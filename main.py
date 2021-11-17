import sys
import time
import itertools
from copy import copy
import networkx as nx
from Graph import Graph
from Solution import Solution


def main():
    if len(sys.argv) < 2:
        print("Please provide an input file!")
        exit()
    
    input = sys.argv[1]
    print(f"Loading input graph from {input}...")
    graph = Graph()
    graph.load_graph_from_file(input)
    
    start = time.process_time()
    task1(graph)
    end = time.process_time()
    print(f"Running time for task1 was: {end - start}")


def construction_heuristic(g : Graph):
    s = Solution()
    
    # TODO: we need to replace this with our own construction heuristic!
    st_1 = nx.algorithms.approximation.steinertree.steiner_tree(g.graph, g.terminal_1, "weight_1")
    st_2 = nx.algorithms.approximation.steinertree.steiner_tree(g.graph, g.terminal_2, "weight_2")
    
    s.key_nodes_1 = [n for n in st_1.nodes if st_1.degree(n) >= 3 and n not in g.terminal_1]
    s.key_nodes_2 = [n for n in st_2.nodes if st_2.degree(n) >= 3 and n not in g.terminal_2]
    
    s.edges_1 = set(st_1.edges)
    s.edges_2 = set(st_2.edges)
    
    s.weight_1 = st_1.size("weight_1")
    s.weight_2 = st_2.size("weight_2")
    
    compute_shared_edges(g, s)
    
    return s
    

def distance_graph(g : Graph, ns : [int], weight : str):
    graph = nx.Graph()
    graph.add_nodes_from(ns)
    
    for s, t in itertools.combinations(ns, 2):
        length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(g.graph, s, t, weight)
        graph.add_edge(s, t, weight=length, path=path)
        
    return graph


def augment_distance_graph(g : Graph, dg : nx.Graph, n : int, weight : str):
    dg.add_node(n)
    for s in dg.nodes():
        if s == n:
            continue
        length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(g.graph, s, n, weight)
        dg.add_edge(s, n, weight=length, path=path)


def rebuild_steiner_tree(g : Graph, dg : nx.Graph, tree : int):
    mst = nx.algorithms.tree.mst.minimum_spanning_tree(dg, weight=f"weight_{tree}_mod")
    total_weight = 0
    
    edges = set()
    
    # remove unnecessary nodes
    terminals = getattr(g, f"terminal_{tree}")
    changed = True
    while changed:
        changed = False
        to_remove = []
        
        for n in mst.nodes:
            if n not in terminals and mst.degree(n) == 1:
                to_remove.append(n)
                changed = True
        
        mst.remove_nodes_from(to_remove)
    
    # translate distance graph edges to original graph edges and compute sum over undiscounted weights
    # (TODO: could be changed to collect discounted weights and then correct based on shared edges --> slightly faster)    
    for (n1,n2) in mst.edges():
        last_n = None
        for n in dg.get_edge_data(n1,n2)["path"]:
            if last_n is None:
                last_n = n
                continue
            edges.add((last_n, n))
            total_weight += g.graph.get_edge_data(last_n, n)[f"weight_{tree}"]
            last_n = n
    
    # TODO: we could have introduced key-nodes which are not in the MST!!!
    return edges, total_weight, [n for n in mst.nodes if mst.degree(n) >= 3 and n not in terminals]


def compute_shared_edges(g : Graph, s : Solution):
    s.weight_s = 0
    shared_edges = s.edges_1.intersection(s.edges_2)
    for e in shared_edges:
        s.weight_s += g.graph.get_edge_data(*e)["weight"]
    s.weight_s *= g.gamma


def compute_add_keynode_next_neighbor(g : Graph, dg : Graph, ns : [int], s : Solution, tree : int, value : int):
    for n in ns:
        # augment distance graph with next node
        augment_distance_graph(g, dg, n, f"weight_{tree}_mod")
        
        # use new distance graph to build steiner tree and create new solution
        edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)

        new_s = copy(s)
        
        setattr(new_s, f"edges_{tree}", edges)
        setattr(new_s, f"weight_{tree}", weight)

        # TODO: We should now eliminate a potential cycle between T1 and T2...
        # e.g. if we just computed T1, remove redundant edges in T2
        # this would make the new solution more acceptable

        # evaluate new solution and return if it is better
        compute_shared_edges(g, new_s)
        new_value = new_s.evaluate()
        
        if new_value < value:
            setattr(new_s, f"key_nodes_{tree}", key_nodes)
            return new_s, value        
        
        # reset the distance graph
        dg.remove_node(n)
    
    return None, None
    
    
def compute_remove_keynode_best_neighbor(g : Graph, dg : Graph, ns : [int], s : Solution, tree : int, value : int):
    best_solution = None
    best_value = value

    for n in ns:
        # create subgraph_view not containing n
        dg_mod = nx.classes.function.subgraph_view(dg, filter_node=lambda ni: ni != n)
        
        # use new distance graph to build steiner tree and create new solution
        edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)

        new_s = copy(s)
        
        setattr(new_s, f"edges_{tree}", edges)
        setattr(new_s, f"weight_{tree}", weight)

        # TODO: We should now eliminate a potential cycle between T1 and T2...
        # e.g. if we just computed T1, remove redundant edges in T2
        # this would make the new solution more acceptable

        # evaluate new solution and store if it is better
        compute_shared_edges(g, new_s)
        new_value = new_s.evaluate()
        
        if new_value < best_value:
            setattr(new_s, f"key_nodes_{tree}", key_nodes)
            best_value = new_value
            best_solution = new_s
        
    return best_solution, best_value
    

def vnd(g : Graph, s : Solution, v : int):
    current = s
    value = v
    
    while True:
        print(f"New best solution:\n{current}")

        start = time.process_time()        
        # discount edges that are already used
        for n1,n2,d in g.graph.edges.data():
            d["weight_1_mod"] = (d["weight_1"] + g.gamma) if (n1,n2) in current.edges_2 else d["weight_1"]
            d["weight_2_mod"] = (d["weight_2"] + g.gamma) if (n1,n2) in current.edges_1 else d["weight_2"]

        # compute the distance graph cores
        nodes_1 = g.terminal_1 + s.key_nodes_1
        nodes_2 = g.terminal_2 + s.key_nodes_2
        dg_1 = distance_graph(g, nodes_1, "weight_1_mod")
        dg_2 = distance_graph(g, nodes_2, "weight_2_mod")
        
        end = time.process_time()
        print(f"Computed modified distance graphs in: {end-start}")
        
        # define nodes to add
        add_nodes = [i for i in range(0, g.abs_V) if i not in nodes_1 and i not in nodes_2]
        add_nodes_1 = [nodes_2, add_nodes]
        add_nodes_2 = [nodes_1, add_nodes]
        
        remove_nodes_1 = current.key_nodes_1
        remove_nodes_2 = current.key_nodes_2
        
        print(add_nodes_1)
        print(add_nodes_2)
        print(remove_nodes_1)
        print(remove_nodes_2)
        
        # compute next/best solutions in neighborhoods
        # TODO: also recompute the trees without changing any key nodes (best improvement)
        
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_1, add_nodes_1[0], current, 1, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_1_a")
            continue
        
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_2, add_nodes_2[0], current, 2, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_1_b")
            continue
        
        new_s, new_v = compute_remove_keynode_best_neighbor(g, dg_1, remove_nodes_1, current, 1, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_2_a")
            continue
            
        new_s, new_v = compute_remove_keynode_best_neighbor(g, dg_2, remove_nodes_2, current, 2, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_2_b")
            continue
        
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_1, add_nodes_1[1], current, 1, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_3_a")
            continue
        
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_2, add_nodes_2[1], current, 2, value)
        if new_s is not None and new_v < value:
            current = new_s
            value = new_v
            print("NH_3_b")
            continue
        
        break
            

def task1(g : Graph):
    init = construction_heuristic(g)
    init_value = init.evaluate()
    
    vnd(g, init, init_value)
    

if __name__ == "__main__":
    main()