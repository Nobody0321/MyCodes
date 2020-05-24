# 使用prim算法查找最小生成树
class Graph:
    def __init__(self):
        self.edges = []
        self.nodes = []
        self.map = None

    def add_edge(self, node1, node2, weight):
        self.edges.append([node1, node2, weight])
        self.add_node(node1)
        self.add_node(node2)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
    
    def gen_map(self):
        num_nodes = max(self.nodes)
        self.map = [[999999] * num_nodes for _ in range(num_nodes)]
        for edge in self.edges:
            self.map[edge[0] - 1][edge[1] - 1] = edge[2]
          
    def prim(self):
        nodes = self.nodes
        w = 0
        nodes.sort()
        selected_node = [nodes[0]]
        candidate_node = nodes[1:]
        while candidate_node != []:
            begin, end, minweight = None, None, 999999
            for i in selected_node:
                for j in candidate_node:
                    if self.map[i - 1][j - 1] < minweight:
                        minweight = self.map[i - 1][j - 1]
                        begin = i
                        end = j
            w += minweight
            selected_node.append(end)
            candidate_node.remove(end)
        return w


graph = Graph()

while True:
    line=input().strip()
    if line == '':
        break
    line = list(map(int, line.split(",")))
    graph.add_edge(line[0], line[1], line[2])

graph.gen_map()

print(graph.prim())
