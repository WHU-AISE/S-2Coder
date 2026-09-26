#!/usr/bin/env python3
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from datetime import datetime

DATASET_PATH = Path('')
CONTEXT_PATH = Path('')
CODEAGENT_PATH = Path('')
MAPCODER_PATH = Path('')
DMCODER_PATH = Path('')
SIMPLERAG_PATH = Path('')
REPOCODER_PATH = Path('')
CODEGRAPH_PATH = Path('')
HYPERAGENT_PATH = Path('')

OUT_XLSX = Path('')
OUT_JSON = Path('')

MAX_CELL_CHARS = 30000


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl_first_by_namespace(path: Path):
    out = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            ns = obj.get('namespace')
            if ns and ns not in out:
                out[ns] = obj
    return out


def norm_status(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {'pass', 'passed', 'success', 'true', '1', 'ok'}:
            return True
        if s in {'error', 'fail', 'failed', 'false', '0', 'wrong'}:
            return False
    return None


def context_status(item):
    th = item.get('test_history')
    if isinstance(th, list) and th and isinstance(th[-1], dict):
        return norm_status(th[-1].get('passed'))
    return None


def codeagent_status(item):
    logs = item.get('tool_logs')
    if isinstance(logs, list):
        flags = [
            log.get('ok')
            for log in logs
            if isinstance(log, dict)
            and log.get('tool') == 'python_repl'
            and isinstance(log.get('ok'), bool)
        ]
        if flags:
            return all(flags)
    return None


def generic_status(item):
    for key in ('Result', 'result', 'is_solved', 'passed', 'test_passed', 'exec_passed'):
        if key in item:
            st = norm_status(item.get(key))
            if st is not None:
                return st
    return None


def to_pass_fail(v):
    if v is True:
        return 'PASS'
    if v is False:
        return 'FAIL'
    return ''


def safe_text(v):
    if v is None:
        return ''
    s = str(v)
    if len(s) > MAX_CELL_CHARS:
        return s[:MAX_CELL_CHARS] + '\n...[TRUNCATED]'
    return s


def col_name(idx):
    # 1-based
    name = ''
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(r, c, value):
    ref = f"{col_name(c)}{r}"
    txt = escape(safe_text(value)).replace('\n', '&#10;')
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{txt}</t></is></c>'


def build_sheet_xml(headers, rows):
    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    parts.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
    parts.append('<sheetViews><sheetView workbookViewId="0"/></sheetViews>')
    parts.append('<sheetFormatPr defaultRowHeight="15"/>')
    parts.append('<sheetData>')

    # header row
    r = 1
    cells = [cell_xml(r, i + 1, h) for i, h in enumerate(headers)]
    parts.append(f'<row r="{r}">' + ''.join(cells) + '</row>')

    for ridx, row in enumerate(rows, start=2):
        cells = [cell_xml(ridx, i + 1, row.get(h, '')) for i, h in enumerate(headers)]
        parts.append(f'<row r="{ridx}">' + ''.join(cells) + '</row>')

    parts.append('</sheetData>')
    parts.append('</worksheet>')
    return ''.join(parts)


def write_xlsx(path: Path, headers, rows):
    sheet_xml = build_sheet_xml(headers, rows)

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="summary" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf/></cellStyleXfs>
  <cellXfs count="1"><xf xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>DevEval 172 Methods Summary</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application>
</Properties>'''

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('docProps/core.xml', core)
        z.writestr('docProps/app.xml', app)
        z.writestr('xl/workbook.xml', workbook)
        z.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        z.writestr('xl/styles.xml', styles)
        z.writestr('xl/worksheets/sheet1.xml', sheet_xml)


def main():
    dataset = load_json(DATASET_PATH)
    context = load_json(CONTEXT_PATH)
    codeagent = load_jsonl_first_by_namespace(CODEAGENT_PATH)
    mapcoder = load_jsonl_first_by_namespace(MAPCODER_PATH)
    dmcoder = load_jsonl_first_by_namespace(DMCODER_PATH)
    simplerag = load_jsonl_first_by_namespace(SIMPLERAG_PATH)
    repocoder = load_jsonl_first_by_namespace(REPOCODER_PATH)
    codegraph = load_jsonl_first_by_namespace(CODEGRAPH_PATH)
    hyperagent = load_jsonl_first_by_namespace(HYPERAGENT_PATH)

    headers = [
        'namespace',
        'repo_name',
        'project_path',
        'original_requirement',
        'context_original_requirement',
        'context_enhanced_requirement',
        'context_initial_divided_tasks',
        'context_approved_tasks_for_retrieval',
        'context_reflection_result',
        'context_selected_libraries',
        'context_selected_snippets_for_gen',
        'context_generated_history',
        'context_test_history',
        'context_error_log',
        'context_all_code',
        'context_all_pass',
        'codeagent_code',
        'codeagent_pass',
        'mapcoder_code',
        'mapcoder_pass',
        'dmcoder_code',
        'dmcoder_pass',
        'simple_rag_code',
        'simple_rag_pass',
        'repocoder_code',
        'repocoder_pass',
        'codegraph_code',
        'codegraph_pass',
        'hyperagent_code',
        'hyperagent_pass',
    ]

    rows = []
    missing = {
        'context': 0,
        'codeagent': 0,
        'mapcoder': 0,
        'dmcoder': 0,
        'simplerag': 0,
        'repocoder': 0,
        'codegraph': 0,
        'hyperagent': 0,
    }

    for item in dataset:
        ns = item.get('namespace')
        if not ns:
            continue

        c = context.get(ns, {}) if isinstance(context, dict) else {}
        ca = codeagent.get(ns)
        mc = mapcoder.get(ns)
        dm = dmcoder.get(ns)
        sr = simplerag.get(ns)
        rp = repocoder.get(ns)
        cg = codegraph.get(ns)
        ha = hyperagent.get(ns)

        if not c: missing['context'] += 1
        if ca is None: missing['codeagent'] += 1
        if mc is None: missing['mapcoder'] += 1
        if dm is None: missing['dmcoder'] += 1
        if sr is None: missing['simplerag'] += 1
        if rp is None: missing['repocoder'] += 1
        if cg is None: missing['codegraph'] += 1
        if ha is None: missing['hyperagent'] += 1

        row = {
            'namespace': ns,
            'repo_name': item.get('repo_name') or c.get('repo_name'),
            'project_path': item.get('project_path') or c.get('project_path'),
            'original_requirement': item.get('all_requirement') or item.get('human_requeriment') or c.get('original_requirement') or '',
            'context_original_requirement': c.get('original_requirement', ''),
            'context_enhanced_requirement': c.get('enhanced_requirement', ''),
            'context_initial_divided_tasks': json.dumps(c.get('initial_divided_tasks', ''), ensure_ascii=False),
            'context_approved_tasks_for_retrieval': json.dumps(c.get('approved_tasks_for_retrieval', ''), ensure_ascii=False),
            'context_reflection_result': json.dumps(c.get('reflection_result', ''), ensure_ascii=False),
            'context_selected_libraries': json.dumps(c.get('selected_libraries', ''), ensure_ascii=False),
            'context_selected_snippets_for_gen': c.get('selected_snippets_for_gen', ''),
            'context_generated_history': json.dumps(c.get('generated_history', ''), ensure_ascii=False),
            'context_test_history': json.dumps(c.get('test_history', ''), ensure_ascii=False),
            'context_error_log': c.get('error_log', ''),
            'context_all_code': c.get('final_code', ''),
            'context_all_pass': to_pass_fail(context_status(c)) if c else '',
            'codeagent_code': (ca or {}).get('prediction', ''),
            'codeagent_pass': to_pass_fail(codeagent_status(ca or {})) if ca else '',
            'mapcoder_code': (mc or {}).get('code', ''),
            'mapcoder_pass': to_pass_fail(generic_status(mc or {})) if mc else '',
            'dmcoder_code': (dm or {}).get('completion', ''),
            'dmcoder_pass': to_pass_fail(generic_status(dm or {})) if dm else '',
            'simple_rag_code': (sr or {}).get('completion', ''),
            'simple_rag_pass': to_pass_fail(generic_status(sr or {})) if sr else '',
            'repocoder_code': (rp or {}).get('completion', ''),
            'repocoder_pass': to_pass_fail(generic_status(rp or {})) if rp else '',
            'codegraph_code': (cg or {}).get('completions', ''),
            'codegraph_pass': to_pass_fail(generic_status(cg or {})) if cg else '',
            'hyperagent_code': (ha or {}).get('completion', ''),
            'hyperagent_pass': to_pass_fail(generic_status(ha or {})) if ha else '',
        }
        rows.append(row)

    write_xlsx(OUT_XLSX, headers, rows)

    stats = {
        'total_rows': len(rows),
        'missing_by_method': missing,
    }
    OUT_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    print('[Done]')
    print('xlsx:', OUT_XLSX)
    print('stats:', OUT_JSON)
    print('rows:', len(rows))
    print('missing:', missing)


if __name__ == '__main__':
    main()
