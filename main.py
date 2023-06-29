import networkx as nx
import random



def prob_of_cluster(g, s):
    S = int(s * g.number_of_nodes())
    random_nodes = random.sample(list(g.nodes), k=S)
    k = 0
    for node in random_nodes:
        num_of_neighbors = len(g.edges(node))
        v = num_of_neighbors * 2 / (num_of_neighbors * (num_of_neighbors - 1) + 0.1)
        k += v
    kl = k / S
    K = g.number_of_edges() * 2 / (g.number_of_nodes() * (g.number_of_nodes() - 1))
    delta = kl / K - 1
    return delta


def get_random_split(g):
    subgraph1_size = g.number_of_nodes() // 2
    subgraph1_nodes = random.sample(list(g.nodes), k=subgraph1_size)
    subgraph2_nodes = [node for node in g.nodes if node not in subgraph1_nodes]

    subgraph1 = nx.Graph([edge for edge in g.edges if edge[0] in subgraph1_nodes and edge[1] in subgraph1_nodes])
    subgraph2 = nx.Graph([edge for edge in g.edges if edge[0] in subgraph2_nodes and edge[1] in subgraph2_nodes])
    print(prob_of_cluster(subgraph1, 0.25))
    print(prob_of_cluster(subgraph2, 0.25))
    u = nx.union(subgraph1, subgraph2, rename=("A", "B"))
    u.add_edge('A0', 'B0')
    u.add_edge('A1', 'B1')
    u.add_edge('A2', 'B2')
    u.add_edge('A3', 'B3')
    u.add_edge('A4', 'B4')
    print(prob_of_cluster(u, 0.25))

    print()
    return subgraph1, subgraph2

def find_node(u, A, B, prev_node):
    A_exp = A.copy()
    B_exp = B.copy()
    added_nodes = set()
    for node in A_exp.nodes:
        if prev_node == node:
            continue
        for edge in u.edges(node):
            if B_exp.has_node(edge[1]) and not A_exp.has_node(edge[1]):
                B_exp.add_edge(*edge)
                added_nodes.add(node)

    nodes_indexes = {}
    for index, node in enumerate(list(B_exp.nodes)):
        nodes_indexes[node] = index

    best_degree = -1
    best_node = ''
    dense = nx.adjacency_matrix(B_exp).todense()
    for added_node in added_nodes:
        s = dense[nodes_indexes[added_node]].sum(1).getA()[0][0]
        if s > best_degree:
            best_degree = s
            print(s)
            best_node = added_node
    return best_node


def optimize_version_with_adj(g, A, B, k=1000):
    best_v = -100000 # prob_of_cluster(A, 0.25) + prob_of_cluster(B, 0.25)
    best_A = A.copy()
    best_B = B.copy()

    prev_best_nodes = (None, None)
    best_node_to_B = None
    best_node_to_A = None
    for i in range(k):
        Ac = best_A.copy()
        Bc = best_B.copy()

        while (best_node_to_B, best_node_to_A) == prev_best_nodes:
            print(best_node_to_B, best_node_to_A)
            print(prev_best_nodes)
            print('+++++')
            best_node_to_B = find_node(g, Ac, Bc, best_node_to_B)
            best_node_to_A = find_node(g, Bc, Ac, best_node_to_A)
            print(best_node_to_B, best_node_to_A)

        prev_best_nodes = (best_node_to_B, best_node_to_A)

        for edge in g.edges:
            if (best_node_to_B == edge[1] and Bc.has_node(edge[0])) \
                    or (best_node_to_B == edge[0] and Bc.has_node(edge[1])):
                Bc.add_edge(*edge)
        Ac.remove_node(best_node_to_B)
        print('to B', best_node_to_B)

        for edge in g.edges:
            if (best_node_to_A == edge[1] and Ac.has_node(edge[0])) \
                    or (best_node_to_A == edge[0] and Ac.has_node(edge[1])):
                Ac.add_edge(*edge)
        Bc.remove_node(best_node_to_A)
        print('to A', best_node_to_A)

        print('------')
        #v = prob_of_cluster(Ac, 1) + prob_of_cluster(Bc, 1)

        U = nx.union(Ac, Bc)

        for edge in g.edges:
            if Bc.has_node(edge[0]) and Ac.has_node(edge[1]) \
                    or Bc.has_node(edge[1]) and Ac.has_node(edge[0]):
                U.add_edge(*edge)

        v = prob_of_cluster(U, 1)
        if v > best_v:
            print(v)
            best_v = v
            best_A = Ac
            best_B = Bc
    print(best_v)
    return best_A, best_B

optimize = optimize_version_with_adj
def count_nodes_with_label(graph, label):
    return sum([1 for node in list(graph.nodes) if label in node])

g = nx.dense_gnm_random_graph(1000, 10000)
h = nx.dense_gnm_random_graph(1000, 10000)

# rename nodes
o = nx.Graph()
A = nx.union(g, o, rename=("A", "O"))
B = nx.union(h, o, rename=("B", "O"))

u = nx.union(A, B)
# u.add_edge('A0', 'B0')

number_of_connections = 10
random_nodes_a = random.sample(list(A.nodes), number_of_connections)
random_nodes_b = random.sample(list(B.nodes), number_of_connections)

for i in range(number_of_connections):
    u.add_edge(random_nodes_a[i], random_nodes_b[i])
    print(random_nodes_a[i], random_nodes_b[i])

u_split_a, u_split_b = get_random_split(u)

print('first subgraph. A:', count_nodes_with_label(u_split_a, 'A'), 'B:', count_nodes_with_label(u_split_a, 'B'))
print('second subgraph. A:', count_nodes_with_label(u_split_b, 'A'), 'B:', count_nodes_with_label(u_split_b, 'B'))

print(prob_of_cluster(A, 0.25))
print(prob_of_cluster(B, 0.25))
print(prob_of_cluster(u, 0.25))
print(prob_of_cluster(u_split_a, 0.25))
print(prob_of_cluster(u_split_b, 0.25))

s1r, s2r = optimize(u, u_split_a, u_split_b, 100)


print('first subgraph A.:', count_nodes_with_label(s1r, 'A'), 'B:', count_nodes_with_label(s1r, 'B'))
print('second subgraph B.:', count_nodes_with_label(s2r, 'A'), 'B:', count_nodes_with_label(s2r, 'B'))
