import sys
import time
import itertools
import random
import math
import bisect
from copy import copy
import networkx as nx
import utils
from Graph import Graph
from Solution import Solution


def main():
    if len(sys.argv) < 4:
        print("Please provide an input file, a value for alpha and a time limit in seconds! For now take 0011.txt and alpha=0.2 for 15 seconds.")
        input = "0011.txt"
        alpha = 0.2
        limit = 5
        exit()

    else:
        input = sys.argv[1]
        alpha = float(sys.argv[2])
        limit = float(sys.argv[3])
        
    random.seed(1234)
    
    print(f"Loading input graph from {input}...")
    graph = Graph()
    graph.load_graph_from_file(input)
    
    start1 = time.process_time()
    s1 = task1(graph)
    end1 = time.process_time()

    start2 = time.process_time()
    s2, time2 = task2(graph, alpha, limit, method="mst")
    end2 = time.process_time()

    #start3 = time.process_time()
    #s3, time3 = task2(graph, alpha, limit, method="path")
    #end3 = time.process_time()
    
    print(f"Running time for task1 was: {end1 - start1}")
    print(f"Best solution was {s1.evaluate()}")
    print(f"Running time for task2 mst was: {end2 - start2}")
    print(f"Best solution was {s2.evaluate()}")
    #print(f"Running time for task2 path was: {end3 - start3}")
    #print(f"Best solution was {s3.evaluate()}")
    
    output1 = input.replace(".txt", "_vnd.txt")
    output2 = input.replace(".txt", "_mst_grasp.txt")
    #output3 = input.replace(".txt", "_path_grasp.txt")
    outputlog = input.replace(".txt", "_log.txt")

    s1.write_solution_to_file(graph, output1)
    s2.write_solution_to_file(graph, output2)
    #s3.write_solution_to_file(graph, output3)
    
    f = open(outputlog, "w")
    f.write(f"Log for instance {input}.\n\n")
    
    f.write(f"VND used alpha=0 and took {end1-start1} seconds.\n")
    f.write(f"{s1}\n\n")

    f.write(f"GRASP with mst used alpha={alpha} and took {end2-start2} seconds. The best instance was found after {time2} seconds.\n")
    f.write(f"{s2}\n\n")

    #f.write(f"GRASP with path used alpha={alpha} and took {end3 - start3} seconds. The best instance was found after {time3} seconds.\n")
    #f.write(f"{s3}")
    
def runall_with_grid(alpha_grid = [0.05,0.1,0.2,0.4,0.7,0.95]):
    files = ["0125.txt", "1331.txt"]
    f = open("girdsearchlog", "w")

    for file in files:
        best_method ,best_alpha, best_s = run_with_grid(file, 600, alpha_grid)
        f.write(f"Log for instance {file}.\n\n")
        f.write(f"best method : {best_method}\n")
        f.write(f"best alpha : {best_alpha}\n")
        f.write(f"best solution with val {best_s.evaluate()} \n {best_s}\n\n")

def run_with_grid(input, limit, alpha_grid):
    if input == "0011.txt":
        limit = 1
    print(f"Loading input graph from {input}...")
    graph = Graph()
    graph.load_graph_from_file(input)

    best_s = None
    best_alpha = None
    best_method = None

    for alpha in alpha_grid:
        start2 = time.process_time()
        s2, time2 = task2(graph, alpha, limit, method="mst")
        end2 = time.process_time()

        start3 = time.process_time()
        s3, time3 = task2(graph, alpha, limit, method="path")
        end3 = time.process_time()
        print(f"With alpha = {alpha}")
        print(f"Running time for task2 mst was: {end2 - start2}")
        print(f"Best solution was {s2.evaluate()}")
        print(f"Running time for task2 path was: {end3 - start3}")
        print(f"Best solution was {s3.evaluate()}")
        if best_s is None:
            if s2.evaluate() < s3.evaluate():
                best_s = s2
                best_alpha = alpha
                best_method = "mst"
            else:
                best_s = s3
                best_alpha = alpha
                best_method = "path"
        else:
            if s2.evaluate() < best_s.evaluate():
                best_s = s2
                best_alpha = alpha
                best_method = "mst"
            elif s3.evaluate() < best_s.evaluate():
                best_s = s3
                best_alpha = alpha
                best_method = "path"
    output = input.replace(".txt", f"best_grasp.txt")
    best_s.write_solution_to_file(graph, output)
    return best_method, best_alpha, best_s


def randomized_mst(dg : nx.Graph, alpha : float):
    key = lambda e: dg.get_edge_data(*e)["weight"]
    
    edges_sorted = sorted(dg.edges, key=key)

    comps = {n : n for n in dg.nodes}
    edges_taken = set()
    for i in range(0, len(dg.nodes) - 1):
        low_end = dg.get_edge_data(*edges_sorted[0])["weight"]
        high_end = dg.get_edge_data(*edges_sorted[-1])["weight"]
        bound = low_end + alpha * (high_end - low_end)
        
        idx_bound = bisect.bisect_right(edges_sorted, bound, key=key)
        
        n1,n2 = edges_sorted[random.randrange(0,idx_bound)]
        c1 = comps[n1]
        c2 = comps[n2]
        if c1 is None and c2 is None:
            comps[n1] = c1
            comps[n2] = c1
        elif c1 is None:
            comps[n1] = c2
        elif c2 is None:
            comps[n2] = c1
        else:
            for n,c in comps.items():
                if c == c2:
                    comps[n] = c1
        
        edges_taken.add(utils.ordered_edge(n1,n2))
        
        edges_sorted = [e for e in edges_sorted if comps[e[0]] != comps[e[1]]]
    
    return nx.classes.function.subgraph_view(dg, filter_edge=lambda n1,n2: utils.ordered_edge(n1,n2) in edges_taken)


def randomized_steiner_tree(g : Graph, tree : int, alpha : float):
    nodes = getattr(g, f"terminal_{tree}").copy()

    alpha_mod = alpha
    #while alpha_mod > 0.0 and len(nodes) < g.abs_V:
    while alpha_mod > 0.0:
        r = random.random()
        if r <= alpha_mod:
            #while True:
                n = random.randrange(0, g.abs_V)
                if n not in nodes:
                    nodes.append(n)
                    alpha_mod *= 0.5
                    #alpha_mod *= 0.9
                    #break
        else:
            break
    
    dg = distance_graph(g, nodes, f"weight_{tree}")
    mst = randomized_mst(dg, alpha)

    terminal = getattr(g, f"terminal_{tree}")
    for t in terminal:
        if not t in mst.nodes:
            print("\n\n\nnot in terminal\n\n\n")

    edges, weight, terminals = rebuild_steiner_tree(g, mst, tree)
    return edges, weight, terminals


def random_path(g: Graph, source, target, weight, cutoff):
    itr = nx.all_simple_paths(g.graph, source, target, cutoff)#cutoff is the search depth not the weight of the graph
    path = next(itr)
    ctr = 0
    path_lst = []
    while True:
        ctr += 1
        path_lst.append(path)
        path = next(itr, None)
        if path is None or ctr > 3000:
            break
    path = random.choice(path_lst)
    length = nx.classes.path_weight(g.graph, path, weight)
    return path, length


def randomized_greedy_steiner_tree(g: Graph, weight, terminal_nodes, alpha = 0.5):#alpha gives the factor how much more nodes the choosen path can have than the shortest path
    pair_shortest_paths = {}
    for s, t in itertools.combinations(terminal_nodes, 2):
        length, path = nx.algorithms.shortest_paths.weighted.bidirectional_dijkstra(g.graph, s, t, weight)
        pair_shortest_paths[(s, t)] = (length, path)

    simple_paths = {}
    edges_for_adding = []
    nodes_for_adding = set()
    for s, t in itertools.combinations(terminal_nodes, 2):
        path, length = random_path(g, s, t, weight, cutoff = len(pair_shortest_paths[(s,t)][1])*(1+alpha))
        simple_paths[(s,t)] = (length,path)
        edges_for_adding.append((s,t,length))
        for p in path:
            nodes_for_adding.add(p)

    dg = nx.Graph()
    dg.add_nodes_from(terminal_nodes)
    dg.add_weighted_edges_from(edges_for_adding)

    mst = nx.algorithms.minimum_spanning_tree(dg)
    steiner_tree = nx.Graph()
    steiner_tree.add_nodes_from(nodes_for_adding)

    edges0 = {}
    for edge in mst.edges:
        for i in range(len(simple_paths[edge][1])-1):
            left = simple_paths[edge][1][i]
            right = simple_paths[edge][1][i+1]
            e = (left,right)
            w_dict = g.graph.get_edge_data(*e)
            if edges0 == {}:
                for w in w_dict:
                    edges0[w] = []
            for w in w_dict:
                edges0[w].append((left,right,w_dict[weight]))
    for w in edges0:
        steiner_tree.add_weighted_edges_from(edges0[w],weight)

    return steiner_tree


def construction_heuristic(g : Graph, alpha = 0, method = "mst"):
    s = Solution()
    if method == "path":
        st_1 = randomized_greedy_steiner_tree(g, "weight_1", g.terminal_1, alpha = alpha)
        st_2 = randomized_greedy_steiner_tree(g, "weight_2", g.terminal_2, alpha = alpha)

        s.key_nodes_1 = [n for n in st_1.nodes if st_1.degree(n) >= 3 and n not in g.terminal_1]
        s.key_nodes_2 = [n for n in st_2.nodes if st_2.degree(n) >= 3 and n not in g.terminal_2]

        s.edges_1 = set(st_1.edges)
        s.edges_2 = set(st_2.edges)

        s.weight_1 = st_1.size("weight_1")
        s.weight_2 = st_2.size("weight_2")

        compute_shared_edges(g, s)

        return s

    if method == "mst":
        s.edges_1, s.weight_1, s.key_nodes_1 = randomized_steiner_tree(g, 1, alpha)
        s.edges_2, s.weight_2, s.key_nodes_2 = randomized_steiner_tree(g, 2, alpha)
    
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
    
    # translate distance graph edges to original graph edges
    edges_weighted = set()
    for (n1,n2) in mst.edges():
        last_n = None
        for n in dg.get_edge_data(n1,n2)["path"]:
            if last_n is None:
                last_n = n
                continue
            w = g.graph.get_edge_data(last_n, n)
            if f"weight_{tree}_mod" in w:
                w = w[f"weight_{tree}_mod"]
            else:
                w = w[f"weight_{tree}"]
            edges.add(utils.ordered_edge(last_n, n))
            edges_weighted.add(utils.ordered_edge_weighted(last_n, n, w))
            last_n = n


    #   For cycle detection build the graph and build the mst unfortunately this is the fastest I could think of for cycle detection.
    #   Because if we use the cycle function of nx we still need to figure out which path is the best to delete which seems like a very daunting task
    #   Therefore cycle deletion is just build the mst and then remove nodes of degree 1 that arent terminal nodes.
    nodes = set()
    graph = nx.Graph()
    for n1,n2 in edges:
        nodes.add(n1)
        nodes.add(n2)
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges_weighted)
    #to remove cycles build the mst of the supposed steiner tree
    st_mst = nx.minimum_spanning_tree(graph,"weight")
    if st_mst.edges != graph.edges:
        #if the edges changed at least one cycle was removed therefore we need to trim the tree further
        #we do the trimming as before
        changed = True
        while changed:
            changed = False
            to_remove = []

            for n in st_mst.nodes:
                if n not in terminals and st_mst.degree(n) == 1:
                    to_remove.append(n)
                    changed = True

            st_mst.remove_nodes_from(to_remove)


    # We could have introduced key-nodes which are not in the MST! Therefore, we have to recompute the key-nodes!
    # Also compute the weight of the new tree.
    node_degree = {}
    total_weight = 0
    edges = set()
    for n1, n2 in st_mst.edges:
        if n1 not in node_degree:
            node_degree[n1] = 0
        if n2 not in node_degree:
            node_degree[n2] = 0
        node_degree[n1] += 1
        node_degree[n2] += 1

        total_weight += g.graph.get_edge_data(n1, n2)[f"weight_{tree}"]
        
        edges.add(utils.ordered_edge(n1,n2))


    return edges, total_weight, [n for n, d in node_degree.items() if d >= 3 and n not in terminals]


def compute_shared_edges(g : Graph, s : Solution):
    s.weight_s = 0
    shared_edges = s.edges_1.intersection(s.edges_2)
    
    for e in shared_edges:
        s.weight_s += g.graph.get_edge_data(*e)["weight"]
    s.weight_s *= g.gamma


def compute_add_keynode_best_neighbor(g : Graph, dg : Graph, ns : [int], s : Solution, tree : int, value : int):
    best_solution = None
    best_value = None

    for n in ns:
        if dg.has_node(n):
            continue
    
        # augment distance graph with next node
        augment_distance_graph(g, dg, n, f"weight_{tree}_mod")
    
        # use new distance graph to build steiner tree and create new solution
        edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)
            
        new_s = copy(s)
        
        setattr(new_s, f"edges_{tree}", edges)
        setattr(new_s, f"weight_{tree}", weight)

        # evaluate new solution and return if it is better
        compute_shared_edges(g, new_s)
        new_value = new_s.evaluate()
        
        if best_value is None or new_value < best_value:
            setattr(new_s, f"key_nodes_{tree}", key_nodes)
            best_value = new_value
            best_solution = new_s
        
        # reset the distance graph
        dg.remove_node(n)
    
    return best_solution, best_value


def compute_add_keynode_next_neighbor(g : Graph, dg : Graph, ns : [int], s : Solution, tree : int, value : int):
    for n in ns:
        if dg.has_node(n):
            continue
    
        # augment distance graph with next node
        augment_distance_graph(g, dg, n, f"weight_{tree}_mod")
    
        # use new distance graph to build steiner tree and create new solution
        edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)
            
        new_s = copy(s)
        
        setattr(new_s, f"edges_{tree}", edges)
        setattr(new_s, f"weight_{tree}", weight)

        # evaluate new solution and return if it is better
        compute_shared_edges(g, new_s)
        new_value = new_s.evaluate()
        
        if new_value < value:
            setattr(new_s, f"key_nodes_{tree}", key_nodes)
            return new_s, new_value
        
        # reset the distance graph
        dg.remove_node(n)
    
    return None, None
     
    
def compute_remove_keynode_best_neighbor(g : Graph, dg : Graph, ns : [int], s : Solution, tree : int, value : int):
    best_solution = None
    best_value = None

    for n in ns:
        # create subgraph_view not containing n
        dg_mod = nx.classes.function.subgraph_view(dg, filter_node=lambda ni: ni != n)
        
        # use new distance graph to build steiner tree and create new solution
        edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)

        new_s = copy(s)
        
        setattr(new_s, f"edges_{tree}", edges)
        setattr(new_s, f"weight_{tree}", weight)

        # evaluate new solution and store if it is better
        compute_shared_edges(g, new_s)
        new_value = new_s.evaluate()
        
        if best_value is None or new_value < best_value:
            setattr(new_s, f"key_nodes_{tree}", key_nodes)
            best_value = new_value
            best_solution = new_s
        
    return best_solution, best_value


def recompute_subtree(g : Graph, dg : Graph, s : Solution, tree : int):
    edges, weight, key_nodes = rebuild_steiner_tree(g, dg, tree)

    new_s = copy(s)
    
    setattr(new_s, f"edges_{tree}", edges)
    setattr(new_s, f"weight_{tree}", weight)
    
    # evaluate new solution and return if it is better
    compute_shared_edges(g, new_s)
    new_value = new_s.evaluate()
    
    setattr(new_s, f"key_nodes_{tree}", key_nodes)
    return new_s, new_value
    

def vnd(g : Graph, s : Solution, v : int, verbose = 0):
    current = s
    value = v
    
    last_tree = None
    nodes_1 = None
    nodes_2 = None
    dg_1 = None
    dg_2 = None
    
    while True:
        if verbose == 2:
            print(f"New best solution:\n{current}")

        start = time.process_time()        
        # discount edges that are already used
        for n1,n2,d in g.graph.edges.data():
            if dg_1 is None:
                d["weight_1_mod"] = (d["weight_1"] + g.gamma) if (n1,n2) in current.edges_2 else d["weight_1"]
            if dg_2 is None:
                d["weight_2_mod"] = (d["weight_2"] + g.gamma) if (n1,n2) in current.edges_1 else d["weight_2"]

        # compute the distance graph cores
        nodes_1 = g.terminal_1 + s.key_nodes_1
        nodes_2 = g.terminal_2 + s.key_nodes_2
        if dg_1 is None:
            dg_1 = distance_graph(g, nodes_1, "weight_1_mod")
        if dg_2 is None:
            dg_2 = distance_graph(g, nodes_2, "weight_2_mod")
        
        end = time.process_time()
        if verbose == 2:
            print(f"Computed modified distance graphs in: {end-start}")
        
        # first, we try to recompute the unchanged tree
        if last_tree is not None:
            new_s, new_v = recompute_subtree(g, dg_2 if last_tree == 1 else dg_1, current, 3 - last_tree)
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 3 - last_tree
                if verbose == 2:
                    print("Recomp start")
                continue

        # define nodes to add/remove
        add_nodes = [i for i in range(0, g.abs_V) if i not in nodes_1 and i not in nodes_2]
        add_nodes_1 = [nodes_2, add_nodes]
        add_nodes_2 = [nodes_1, add_nodes]
        
        remove_nodes_1 = current.key_nodes_1
        remove_nodes_2 = current.key_nodes_2
        
        # intermediate solution for 2 stage adapt
        """
        best_s_i = None
        best_v_i = None
        best_tree = None
        """
        
        # compute next/best solutions in neighborhoods
        # we start with the add-keynode neighborhood on add_nodes[0]
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_1, add_nodes_1[0], current, 1, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 1
                dg_2 = None
                if verbose == 2:
                    print("NH_1_a")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 1
            """
            
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_2, add_nodes_2[0], current, 2, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 2
                dg_1 = None
                if verbose == 2:
                    print("NH_1_b")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 2
            """
            
        # if nothing is found, move on to the remove-keynode neighborhood
        new_s, new_v = compute_remove_keynode_best_neighbor(g, dg_1, remove_nodes_1, current, 1, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 1
                dg_2 = None
                if verbose == 2:
                    print("NH_2_a")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 1
            """
            
        new_s, new_v = compute_remove_keynode_best_neighbor(g, dg_2, remove_nodes_2, current, 2, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 2
                dg_1 = None
                if verbose == 2:
                    print("NH_2_b")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 2
            """
            
        # if still nothing is found, retry with add-keynode neighborhood on add_nodes[1]
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_1, add_nodes_1[1], current, 1, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 1
                dg_2 = None
                if verbose == 2:
                    print("NH_3_a")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 1
            """
            
        new_s, new_v = compute_add_keynode_next_neighbor(g, dg_2, add_nodes_2[1], current, 2, value)
        if new_s is not None:
            if new_v < value:
                current = new_s
                value = new_v
                last_tree = 2
                dg_1 = None
                if verbose == 2:
                    print("NH_3_b")
                continue
            """
            elif best_v_i is None or best_v_i > new_v:
                best_s_i = new_s
                best_v_i = new_v
                best_tree = 2
            """
            
        # now choose best_s and recompute unchanged tree as a last attempt to improve
        """
        if best_s_i is not None:
            for n1,n2,d in g.graph.edges.data():
                if best_tree == 2:
                    d["weight_1_mod"] = (d["weight_1"] + g.gamma) if (n1,n2) in current.edges_2 else d["weight_1"]
                if best_tree == 1:
                    d["weight_2_mod"] = (d["weight_2"] + g.gamma) if (n1,n2) in current.edges_1 else d["weight_2"]

            if best_tree == 2:
                dg_1 = distance_graph(g, nodes_1, "weight_1_mod")
            if best_tree == 1:
                dg_2 = distance_graph(g, nodes_2, "weight_2_mod")
            
            new_s, new_v = recompute_subtree(g, dg_2 if best_tree == 1 else dg_1, current, 3 - best_tree)
            if new_v < value:
                current = new_s
                value = new_v
                
                last_tree = 3 - best_tree
                if last_tree == 1:
                    dg_2 = None
                if last_tree == 2:
                    dg_1 = None

                if verbose == 2:
                    print("Recomp end")
                continue
        """
        
        break
        
    return current

   
def grasp(g : Graph, alpha = 0.5, max_running_time = 15, method = "mst", verbose = 0):
    best = None
    best_value = None
    best_time = None

    start = time.process_time()
    running_time = 0
    while running_time < max_running_time:
        current = construction_heuristic(g, 0 if best is None else alpha, method = method)
        value = current.evaluate()
        
        if verbose == 2:
            print("\nStarting VND")
            
        current = vnd(g, current, value, verbose)
        value = current.evaluate()
        
        if best_value is None or value < best_value:
            best = current
            best_value = value
            best_time = time.process_time()

        running_time = time.process_time() - start

    return best, best_time - start


def task1(g : Graph):
    init = construction_heuristic(g)
    init_value = init.evaluate()
    return vnd(g, init, init_value, verbose = 0)


def task2(g : Graph, alpha : float, limit : int, method : str):
    return grasp(g, alpha, limit, method = method, verbose = 0)


if __name__ == "__main__":
    main()
    #runall_with_grid()