from tree_sitter import Language, Parser
import tree_sitter_python as py_ts
import tree_sitter_java as java_ts


def load_TS(lanuage):
    if lanuage == "python":
        TS_LANAGUGE = Language(py_ts.language())
    elif lanuage == "java":
        TS_LANAGUGE = Language(java_ts.language())
    else:
        raise ValueError("Unsupported language")
    
    parser =Parser(TS_LANAGUGE)
    return parser

class ControlFlowNode:
    def __init__(self, node_id, kind, name):
        self.id = node_id
        self.kind = kind
        self.name = name

class ControlFlowGraphBuilder:
    def __init__(self):
        self.current_id = 0
        self.nodes = {}
        self.edges = []
        self.last_id = None

    def add_node(self, kind, name):
        node_id = self.current_id
        self.current_id += 1
        self.nodes[node_id] = ControlFlowNode(node_id, kind, name)
        if self.last_id is not None:
            self.edges.append({'between': [self.last_id, node_id], 'edgeType': 'Next'})
        self.last_id = node_id
        return node_id

    def traverse_tree(self, node, source_code):
        kind = node.type
        name = source_code[node.start_byte:node.end_byte].decode('utf-8')
        self.add_node(kind, name)

        for child in node.children:
            self.traverse_tree(child, source_code)

def build_cfg_from_code(code, language):
    try:
        parser = load_TS(language)
        codebytes = bytes(code, "utf8")
        codebytes_list = codebytes.split(b'\n')
        tree = parser.parse(codebytes)
        root = tree.root_node

        cfg_builder = ControlFlowGraphBuilder()
        cfg_builder.traverse_tree(root, code.encode('utf-8'))

        node_list = [{'ID': node.id, 'kind': node.kind, 'name': node.name} 
                        for node_id, node in cfg_builder.nodes.items()]
        edge_list = [edge for edge in cfg_builder.edges]
        return node_list, edge_list
    except Exception as e:
        print(f"Error encountered: {e}. Skipping this code segment.")
        return None
    
class DataFlowNode:
    def __init__(self, node_id, name, kind):
        self.id = node_id
        self.name = name
        self.kind = kind

class DataFlowEdge:
    def __init__(self, src, dest, edge_type):
        self.between = [src, dest]
        self.edgeType = edge_type

class DataFlowGraphBuilder:
    def __init__(self):
        self.current_id = 0
        self.nodes = {}
        self.edges = []
        self.temp_counter = 0

    def new_temp(self):
        self.temp_counter += 1
        return f"temp_{self.temp_counter}"

    def add_node(self, name, kind):
        node_id = self.current_id
        self.current_id += 1
        self.nodes[node_id] = DataFlowNode(node_id, name, kind)
        return node_id

    def add_edge(self, src, dest, edge_type):
        self.edges.append(DataFlowEdge(src, dest, edge_type))

    def traverse_tree(self, node, source_code):
        kind = node.type
        name = source_code[node.start_byte:node.end_byte].decode('utf-8')
        current_id = self.add_node(name, kind)

        for child in node.children:
            child_id = self.traverse_tree(child, source_code)
            self.add_edge(child_id, current_id, 'Child')

        return current_id

def build_dfg_from_code(code, language):
    try:
        parser = load_TS(language)
        codebytes = bytes(code, "utf8")
        codebytes_list = codebytes.split(b'\n')
        tree = parser.parse(codebytes)
        root = tree.root_node
        dfg_builder = DataFlowGraphBuilder()
        dfg_builder.traverse_tree(root, code.encode('utf-8'))

        node_list = [{'ID': node.id, 'name': node.name, 'kind': node.kind} 
                     for node_id, node in dfg_builder.nodes.items()]
        edge_list = [{'between': edge.between, 'edgeType': edge.edgeType} 
                     for edge in dfg_builder.edges]

        return node_list, edge_list
    except Exception as e:
        print(f"Error encountered: {e}. Skipping this code segment.")
        return None           


def build_csg_from_code(code, language):
    try:
        # 获取 CFG 和 DFG
        cfg_nodes, cfg_edges = build_cfg_from_code(code, language)
        dfg_nodes, dfg_edges = build_dfg_from_code(code, language)

        if cfg_nodes is None or dfg_nodes is None:
            raise ValueError("Failed to generate CFG or DFG.")

        # 合并节点
        csg_nodes = {node['ID']: node for node in cfg_nodes}
        max_cfg_id = max(csg_nodes.keys(), default=-1)
        for node in dfg_nodes:
            if node['ID'] not in csg_nodes:
                # 确保 DFG 的节点 ID 不与 CFG 冲突
                node['ID'] += max_cfg_id + 1
                csg_nodes[node['ID']] = node

        # 合并边
        csg_edges = cfg_edges + [
            {'between': [edge['between'][0] + max_cfg_id + 1, edge['between'][1] + max_cfg_id + 1], 
             'edgeType': edge['edgeType']} 
            for edge in dfg_edges
        ]

        # 转换为列表形式
        csg_node_list = list(csg_nodes.values())
        csg_edge_list = csg_edges

        return csg_node_list, csg_edge_list
    except Exception as e:
        print(f"Error encountered while building CSG: {e}")
        return None


if __name__ == "__main__":
    code = """
    for i in range(1, 3):
        x += 1
        sum += x
    """
    # print(build_cfg_from_code(code, "python"))
    # print(build_dfg_from_code(code, "python"))
    print(build_csg_from_code(code, "python"))