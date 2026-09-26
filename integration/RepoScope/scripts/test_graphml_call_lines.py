#!/usr/bin/env python3
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def load_call_lines(call_line_dir, repo_name, graphml_path, repo_root_path):
    call_lines_map_by_ids = {}
    call_lines_map_by_plain = {}
    graphml_base = os.path.splitext(os.path.basename(graphml_path))[0]
    candidates = [
        os.path.join(call_line_dir, f"{repo_name}.calls.json"),
        os.path.join(call_line_dir, f"{graphml_base}.calls.json"),
        os.path.join(call_line_dir, f"{repo_name}.json"),
        os.path.join(call_line_dir, f"{graphml_base}.json"),
    ]
    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break
    if not found and os.path.isdir(call_line_dir):
        for c in os.listdir(call_line_dir):
            if graphml_base in c:
                found = os.path.join(call_line_dir, c)
                break
    if not found:
        return call_lines_map_by_ids, call_lines_map_by_plain

    try:
        with open(found, 'r', encoding='utf8') as fh:
            data = json.load(fh)
            # handle nested file -> node_key -> [calls]
            if isinstance(data, dict) and not ('edges_calls' in data) and all(isinstance(v, dict) for v in data.values()):
                for file_key, node_map in data.items():
                    plain_file = os.path.join(str(repo_root_path), file_key)
                    if not isinstance(node_map, dict):
                        continue
                    for node_key, calls in node_map.items():
                        caller_id = str(node_key)
                        if isinstance(calls, dict):
                            for callee, lines in calls.items():
                                if lines is None:
                                    continue
                                if isinstance(lines, int):
                                    lines = [lines]
                                try:
                                    lst = list(map(int, lines))
                                except Exception:
                                    continue
                                call_lines_map_by_ids[(caller_id, str(callee))] = lst
                                call_lines_map_by_plain[(plain_file, str(callee))] = lst
                        elif isinstance(calls, list):
                            for call in calls:
                                if not isinstance(call, dict):
                                    continue
                                callee = call.get('callee') or call.get('callee_node') or call.get('callee_id') or call.get('callee_plain')
                                line = call.get('line') or call.get('call_line') or call.get('call_lines') or call.get('lines')
                                if callee is None or line is None:
                                    continue
                                if isinstance(line, int):
                                    lst = [int(line)]
                                else:
                                    try:
                                        lst = list(map(int, line))
                                    except Exception:
                                        continue
                                call_lines_map_by_ids[(caller_id, str(callee))] = lst
                                call_lines_map_by_plain[(plain_file, str(callee))] = lst
            else:
                # flatten lists or edges_calls
                records = []
                if isinstance(data, dict) and 'edges_calls' in data:
                    records = data['edges_calls']
                elif isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = [data]
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    caller = rec.get('caller_node') or rec.get('caller') or rec.get('caller_id') or rec.get('caller_plain')
                    callee = rec.get('callee_node') or rec.get('callee') or rec.get('callee_id') or rec.get('callee_plain')
                    lines = rec.get('call_lines') or rec.get('call_lines_list') or rec.get('lines') or rec.get('call_line') or []
                    if lines is None:
                        lines = []
                    if isinstance(lines, int):
                        lines = [lines]
                    try:
                        lst = list(map(int, lines))
                    except Exception:
                        lst = []
                    if caller and callee and lst:
                        call_lines_map_by_ids[(str(caller), str(callee))] = lst
                        call_lines_map_by_plain[(str(caller), str(callee))] = lst
    except Exception as e:
        print('failed loading call lines', e)
    return call_lines_map_by_ids, call_lines_map_by_plain


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--graphml', required=True)
    p.add_argument('--repo-root', required=True)
    p.add_argument('--calllinedir', required=True)
    p.add_argument('--repo-name', required=False)
    p.add_argument('--chain-weight', type=float, default=1.0)
    p.add_argument('--call-weight', type=float, default=2.0)
    p.add_argument('--output', required=False, help='Write JSON chain map to this path')
    args = p.parse_args()

    graphml = args.graphml
    repo_root = args.repo_root
    calllinedir = args.calllinedir
    repo_name = args.repo_name or os.path.splitext(os.path.basename(graphml))[0]

    tree = ET.parse(graphml)
    root = tree.getroot()

    # build id->plain mapping (same heuristic as parser)
    id_to_plain = {}
    for n in root.findall('.//{*}node'):
        nid = n.get('id')
        data = n.find('{*}data')
        label = data.text if data is not None else None
        if nid is None or '::' not in nid:
            continue
        parts = nid.split('::')
        is_function = len(parts) > 3 and parts[3].isdigit() and not parts[2].isdigit()
        is_class = len(parts) > 2 and parts[2].isdigit()
        if not (is_function or is_class):
            continue
        nid_fs = nid.replace('::', os.sep)
        path = os.path.join(str(repo_root), nid_fs)
        id_to_plain[nid] = path

    call_lines_ids, call_lines_plain = load_call_lines(calllinedir, repo_name, graphml, repo_root)

    source_to_targets = defaultdict(list)
    call_edges = 0
    for e in root.findall('.//{*}edge'):
        src = e.get('source')
        tgt = e.get('target')
        data = e.find('{*}data')
        data_text = data.text.strip() if data is not None and data.text is not None else ''
        if data_text != 'call':
            continue
        if src in id_to_plain and tgt in id_to_plain:
            call_edges += 1
            s = id_to_plain[src]
            t = id_to_plain[tgt]
            # prefer JSON lines
            start_line = 0
            if (src, tgt) in call_lines_ids:
                lst = call_lines_ids[(src, tgt)]
                start_line = min(lst) if lst else 0
            else:
                src_plain = id_to_plain.get(src)
                tgt_plain = id_to_plain.get(tgt)
                # try by plain pair
                if src_plain and tgt_plain and (src_plain, tgt_plain) in call_lines_plain:
                    lst = call_lines_plain[(src_plain, tgt_plain)]
                    start_line = min(lst) if lst else 0
                # try by callee name
                if start_line == 0:
                    callee_name = os.path.basename(tgt_plain) if tgt_plain else os.path.basename(tgt)
                    if (src, callee_name) in call_lines_ids:
                        lst = call_lines_ids[(src, callee_name)]
                        start_line = min(lst) if lst else 0
                    elif (src_plain, callee_name) in call_lines_plain:
                        lst = call_lines_plain[(src_plain, callee_name)]
                        start_line = min(lst) if lst else 0
            if start_line == 0:
                for part in tgt.split('::'):
                    if part.isdigit():
                        start_line = int(part)
                        break
            source_to_targets[src].append((tgt, start_line))

    print('nodes from graphml:', len(id_to_plain))
    print('call edges found:', call_edges)

    chain_weight = args.chain_weight
    chain_map = defaultdict(lambda: defaultdict(float))
    for src_id, targets in source_to_targets.items():
        targets_sorted = sorted(targets, key=lambda x: x[1])
        for i in range(len(targets_sorted)):
            prev_id, prev_line = targets_sorted[i]
            for j in range(i + 1, len(targets_sorted)):
                cur_id, cur_line = targets_sorted[j]
                delta = max(1, cur_line - prev_line)
                chain_map[prev_id][cur_id] += chain_weight / delta

    # prepare output
    total_pairs = sum(len(m) for m in chain_map.values())
    sample = []
    cnt = 0
    for prev, m in chain_map.items():
        for cur, val in m.items():
            sample.append({'prev': prev, 'cur': cur, 'value': val})
            cnt += 1
            if cnt >= 20:
                break
        if cnt >= 20:
            break

    out = {
        'nodes_from_graphml': len(id_to_plain),
        'call_edges_found': call_edges,
        'total_chain_pairs': total_pairs,
        'sample': sample,
        'config': {'chain_weight': args.chain_weight, 'call_weight': args.call_weight}
    }
    if args.output:
        try:
            import json
            with open(args.output, 'w', encoding='utf8') as fh:
                # serialize chain_map as nested dict
                serial = {k: {kk: v for kk, v in mm.items()} for k, mm in chain_map.items()}
                fh.write(json.dumps({'meta': out, 'chain': serial}, ensure_ascii=False, indent=2))
            print('wrote', args.output)
        except Exception as e:
            print('failed writing output', e)
    else:
        import json
        print(json.dumps({'meta': out, 'sample': sample}, ensure_ascii=False, indent=2))
