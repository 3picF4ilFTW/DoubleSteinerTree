import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import utils
from copy import deepcopy


class Graph:
    def __init__(self):
        None

    def load_graph_from_file(self, filename):
        self.graph = nx.Graph()
        with open(filename) as f:  # opening the txt file
            lines = f.readlines()
        for i in range(len(lines)):  # deleting unnecessary information in the string and splitting into list
            lines[i] = lines[i].replace("\n", "").split(" ")
        self.abs_V = int(lines[0][1])
        self.abs_E = int(lines[0][2])
        self.abs_T_1 = int(lines[0][3])
        self.abs_T_2 = int(lines[0][4])
        self.alpha_1 = int(lines[1][1])
        self.alpha_2 = int(lines[1][2])
        self.gamma = int(lines[1][3])

        vertices = []
        for i in range(self.abs_V):
            vertices.append(i)

        edges = []
        self.terminal_1 = []
        self.terminal_2 = []
        for line in lines:
            if line[0] == "E":
                edges.append(utils.ordered_edge_weighted(int(line[1]), int(line[2]), int(line[3])))  # collects edges out of the file with weight
            if line[0] == "T1":
                self.terminal_1.append(int(line[1]))  # collects the Terminals1 out of file
            if line[0] == "T2":
                self.terminal_2.append(int(line[1]))

        self.graph.add_nodes_from(vertices)
        self.graph.add_weighted_edges_from(edges)

        for n1,n2,d in self.graph.edges.data():
            d["weight_1"] = self.alpha_1 * d["weight"]
            d["weight_2"] = self.alpha_2 * d["weight"]
        
        #self.graph_shortest_paths = dict(nx.algorithms.shortest_paths.all_pairs_dijkstra(self.graph))
        # returns a dict of tuple of dict with each pair, Unfortunately this dict is ordered a bit weirdly as the first element in the dict is the source node.
        # The elements in the tuple then are 0 for the length and 1 for the path, The elements in the final dict are the target node
        # takes 26 seconds for the longest instance thus is very expensive but it decreases the length of construct distance graph to almost 0.


    def adjust_weights_in_dict_and_graph(self, graph: nx.Graph, shortest_path_dict: dict, previous_key_edges: list,
                                         new_key_edges: list, alpha, gamma):
        # TODO I think we have to recalculate all shortest paths after we adjust even a single weight in a graph.
        # TODO e.g. if a weight changes from 100 to 1 it might make it possible for all subsequent paths to use this edge and it is impossible to tell which paths might change
        None

    def get_from_short_path_dict(self, short_path_dict, source, target,
                                 path_or_length):  # because of the weird ordering i wrote an access method
        if path_or_length == "length":
            return short_path_dict[source][0][target]
        else:
            return short_path_dict[source][1][target]

    def construct_distance_graph(self, nodes: list,
                                 shortest_path_dictionary: dict = None):  # constructs the distance graph for a list of nodes
        if shortest_path_dictionary is None:
            shortest_path_dictionary = self.graph_shortest_paths
        short_path_list = []
        # this saves the shortest path between two nodes as a list as well as the lenght of the path
        for i in range(len(nodes)):
            for j in range(i):
                if i == j:
                    continue
                else:
                    short_path_list.append((nodes[j], nodes[i],
                                            self.get_from_short_path_dict(shortest_path_dictionary, nodes[j], nodes[i],
                                                                          "length")))
                    # this constructs the list described above
        dist_graph = nx.Graph()
        dist_graph.add_nodes_from(nodes)
        dist_graph.add_weighted_edges_from(short_path_list)
        return dist_graph

    def get_key_nodes_from_steiner_tree(self, steiner_tree: nx.Graph):
        key_nodes = []
        for node in steiner_tree:
            if steiner_tree.degree[node] >= 3:
                key_nodes.append(node)
        return key_nodes

    def get_key_edges_from_steiner_tree_and_key_nodes(self, key_nodes: list, terminal_nodes: list,
                                                      shortest_path_dictionary: dict):
        forbidden_list = key_nodes + terminal_nodes
        key_edges = []
        for node1 in forbidden_list:
            for node2 in forbidden_list:
                if node1 == node2 or (node2, node1) in key_edges:
                    continue
                pa = self.get_from_short_path_dict(shortest_path_dictionary, node1, node2, "path")
                flag = 1
                for i in range(1, len(pa) - 1):
                    if pa[i] in forbidden_list:
                        flag = 0
                        break
                if flag == 1:
                    key_edges.append((node1, node2))
        return key_edges

    def total_weight_in_a_graph(self, graph: nx.Graph):
        w = 0
        att = nx.get_edge_attributes(graph, "weight")
        for edge in graph.edges:
            w += att[edge]
        return w

    def construct_steiner_tree_from_key_nodes(self, key_nodes: list, terminal_nodes: list,
                                              shortest_path_dictionary: dict = None, verbose=False):
        if shortest_path_dictionary is None:
            shortest_path_dictionary = self.graph_shortest_paths

        if (any([True for i in key_nodes if i in terminal_nodes])):
            print("key_nodes and terminal_nodes are not disjunct something went wrong")
            return

        dist_graph = self.construct_distance_graph(key_nodes + terminal_nodes, shortest_path_dictionary)
        mst = nx.algorithms.minimum_spanning_tree(dist_graph, "weight")
        steiner_nodes = []

        for edge in mst.edges:  # calculates the total weight of the steiner tree
            steiner_nodes += self.get_from_short_path_dict(shortest_path_dictionary, edge[0], edge[1],
                                                           "path")  # calculates the steiner nodes of the mst

        steiner_nodes = list(set(steiner_nodes))  # makes the steiner nodes unique

        steiner_tree = nx.Graph()
        steiner_tree.add_nodes_from(steiner_nodes)
        for edge in mst.edges:  # constructs the steiner_tree
            for i in range(len(self.get_from_short_path_dict(shortest_path_dictionary, edge[0], edge[1], "path")) - 1):
                start_node = self.get_from_short_path_dict(shortest_path_dictionary, edge[0], edge[1], "path")[i]
                end_node = self.get_from_short_path_dict(shortest_path_dictionary, edge[0], edge[1], "path")[i + 1]
                w = self.get_from_short_path_dict(shortest_path_dictionary, start_node, end_node, "length")
                steiner_tree.add_edge(start_node, end_node, weight=w)

        total_weight = self.total_weight_in_a_graph(steiner_tree)
        key_nodes = self.get_key_nodes_from_steiner_tree(steiner_tree)
        # TODO this could probably be improved by not constructing the steiner tree dict as it should be possible otherwise
        # TODO The runtime cost of this is not very high thou thus I have let it stay for now
        steiner_short_path_dict = dict(nx.algorithms.shortest_paths.all_pairs_dijkstra(steiner_tree))
        key_edges = self.get_key_edges_from_steiner_tree_and_key_nodes(key_nodes, terminal_nodes,
                                                                       steiner_short_path_dict)
        if verbose:
            print("time for steiner tree dict to be constructed")
            print(end - start)
            print(f"terminal nodes{terminal_nodes}")
            print(f"weight {total_weight}")
            print(f"key_nodes: {key_nodes}")
            print(f"key_edges: {key_edges}")
        return ((steiner_tree, total_weight, key_nodes, key_edges))



    def draw(self, graph=None):
        if graph is None:
            nx.draw_networkx(self.graph, with_labels=True)
            plt.draw()
            plt.show()
        elif nx.Graph == type(graph):
            nx.draw_networkx(graph, with_labels=True)
            plt.draw()
            plt.show()
        else:
            print("wrong input")

    def draw_trees(self):
        edges = list(self.mse())
        nx.draw_networkx(self.graph, with_labels=True, edgelist=edges)
        plt.draw()
        plt.show()

        nx.draw_networkx(self.st(), with_labels=True)
        plt.draw()
        plt.show()


def test_1():
    start = time.time()
    # g.st() #runs about 30s for the longest input
    end = time.time()
    print(f"steiner tree runtime: {end - start}")

    start = time.time()
    g.mst()
    end = time.time()
    print(f"minimum spanning tree runtime: {end - start}")

    start = time.time()
    g.mse()
    end = time.time()
    print(f"minimum spanning edges runtime: {end - start}")

    start = time.time()
    g.shortest_path(0, 1)
    end = time.time()
    print(f"shortest path runtime: {end - start}")

    start = time.time()
    g.draw()
    end = time.time()
    print(f"drawing runtime: {end - start}")

    start = time.time()
    g.draw_trees()
    end = time.time()
    print(f"drawing runtime: {end - start}")


def test_2():  # construct distance graph test
    g2 = g.construct_distance_graph(g.graph, [1, 2, 3, 4])
    print(type(g2))
    m = nx.algorithms.maximum_spanning_tree(g2, "weight")
    g.draw(m)
    g.draw(g2)


def test_3(g: Graph):  # test for construct_steiner_tree_from_key_nodes
    start = time.time()
    np.random.seed(123)
    g.construct_steiner_tree_from_key_nodes([],
                                            g.terminal_1)  # This takes 1ms for the biggest instance thus is relatively fast
    end = time.time()
    print(f"construct steiner tree from key node takes {end - start}")


def test_4(g: Graph):  # test for construct from key_edges
    start = time.time()
    np.random.seed(123)
    g.construct_steiner_tree_from_key_nodes_and_key_edges([], [(1, 2)], g.terminal_1)
    end = time.time()
    print(f"construct steiner tree from key node takes {end - start}")


if __name__ == "__main__":
    g = Graph()
    start = time.time()
    g.load_graph_from_file("1331.txt")
    end = time.time()
    print(f"loading runtime: {end - start}")
    
    """
    start = time.time()
    data = nx.get_edge_attributes(g.g1, "weight")
    end = time.time()
    print(data)
    print(f"test: {end - start}")
    
    data2 = deepcopy(data)
    start = time.time()
    nx.set_edge_attributes(g.g1, data2, "weight_modified")
    end = time.time()
    print(f"test: {end - start}")
    
    start = time.time()
    nx.set_edge_attributes(g.g1, data2, "weight_modified")
    for i in g.terminal_1:
        for j in g.terminal_1:
            length, path = nx.single_source_dijkstra(g.g1, i, j)
    end = time.time()
    print(f"test: {end - start}")
    
    
    data1 = nx.get_edge_attributes(g.g1, "weight")
    data2 = nx.get_edge_attributes(g.g1, "weight")
    
    print(data1[(1260, 1315)])
    print(data2[(1260, 1315)])
    
    data2[(1260, 1315)] -= 2
    
    print(data1[(1260, 1315)])
    print(data2[(1260, 1315)])
    """
    
    # test_4(g)
    # test_1()
    # test_2()
    # test_3(g)
