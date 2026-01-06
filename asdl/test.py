
# from asdl import ASDLGrammar
# from lang.java.java_asdl_helper import *
# from lang.java.java_transition_system import *
# from hypothesis import *
# import javalang.parse

# def generate_java_xml_asdl(code):
#     asdl_text = open('/home/user/nsy/Code/RSSCoder/asdl/lang/java/java_asdl.simplified.txt').read()
#     grammar = ASDLGrammar.from_text(asdl_text)
#     java_ast = javalang.parse.parse(code)

#     # convert the python AST into general-purpose ASDL AST used by tranX
#     asdl_ast = java_ast_to_asdl_ast(java_ast, grammar)
#     print('String representation of the ASDL AST: \n%s' % asdl_ast.to_string())
#     print('Size of the AST: %d' % asdl_ast.size)

#     asdl_ast_xml = java_ast_to_asdl_ast_xml(java_ast, grammar)
#     print('XML representation of the ASDL AST: \n%s' % asdl_ast_xml)

#     return asdl_ast.to_string(), asdl_ast_xml

# code = """
# public class Example {
#     @SuppressWarnings("unchecked")
#     public static Supplier<String> createStringSupplier(int start)
#     {
#         int[] container = new int[] { start };
#         return (Supplier<String> & Serializable) () -> String.valueOf(container[0]++);
#     }
# }
# """


# import javalang

# generate_java_xml_asdl(code)



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

def remove_comments_java(source: str, language: str) -> str:
    parser = load_TS(language)
    codebytes = bytes(source, "utf8")
    codebytes_list = codebytes.split(b'\n')
    tree = parser.parse(codebytes)
    root = tree.root_node

    comment_points = []

    def traverse(node):
        if node.type in ['block_comment', 'line_comment']:
            start_row, start_col = node.start_point
            end_row, end_col = node.end_point
            comment_points.append((start_row, start_col, end_row, end_col))
        
        
        for child in node.children:
            traverse(child)

    traverse(root)

    result = delete_content_with_markers(codebytes_list, comment_points)
    return delete_block_line(b'\n'.join(result).decode("utf-8"))


def remove_comments_py(source: str, language: str) -> str:
    parser = load_TS(language)
    codebytes = bytes(source, "utf8")
    codebytes_list = codebytes.split(b'\n')
    tree = parser.parse(codebytes)
    root = tree.root_node

    comment_points = []


    def traverse(node):
        if node.type == 'comment':
            start_row, start_col = node.start_point
            end_row, end_col = node.end_point
            comment_points.append((start_row, start_col, end_row, end_col))
        
        if node.type == 'string':
            if len(node.parent.children) == 1:
                start_row, start_col = node.start_point
                end_row, end_col = node.end_point
                comment_points.append((start_row, start_col, end_row, end_col))
        
        for child in node.children:
            traverse(child)
    
    traverse(root)

    result = delete_content_with_markers(codebytes_list, comment_points)
    return delete_block_line(b'\n'.join(result).decode("utf-8"))


def delete_content_with_markers(content_list, position_list):
    """
    使用标记删除内容并重建列表
    
    Args:
        content_list: 包含字符串bytes转换后的列表
        position_list: 位置列表，每个元素为(开始行号, 开始列号, 结束行号, 结束列号)
    """
    # 转换为字符串列表
    str_list = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in content_list]
    
    # 创建标记数组
    markers = [[True] * len(line) for line in str_list]
    
    # 标记需要删除的区域
    for start_row, start_col, end_row, end_col in position_list:
        if start_row >= len(str_list) or end_row >= len(str_list):
            continue
            
        # 标记起始行
        if start_row == end_row:
            for col in range(start_col, min(end_col, len(str_list[start_row]))):
                markers[start_row][col] = False
        else:
            # 跨行标记
            # 起始行
            for col in range(start_col, len(str_list[start_row])):
                markers[start_row][col] = False
            
            # 中间行
            for row in range(start_row + 1, end_row):
                if row < len(str_list):
                    markers[row] = [False] * len(str_list[row])
            
            # 结束行
            for col in range(0, min(end_col, len(str_list[end_row]))):
                markers[end_row][col] = False
    
    # 重建列表
    result = []
    for i, line in enumerate(str_list):
        if i < len(markers):
            new_line = ''.join([char for j, char in enumerate(line) if j < len(markers[i]) and markers[i][j]])
            result.append(new_line.encode('utf-8'))
        else:
            result.append(line.encode('utf-8'))
    
    return result
   
def delete_block_line(code):
    lines = [line.rstrip() for line in code.splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip() != "")
    return cleaned


python_code = '''
    a = "cwscwsc csd"
    b = a.split()
    c = getabc()
'''


parser = load_TS("python")
codebytes = bytes(python_code, "utf8")
codebytes_list = codebytes.split(b'\n')
tree = parser.parse(codebytes)
root = tree.root_node
print(root)