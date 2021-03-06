from Graph import Graph

class Solution:
    def __init__(self):
        self.key_nodes_1 = []
        self.key_nodes_2 = []
        
        self.edges_1 = set()
        self.edges_2 = set()
        
        self.weight_1 = 0
        self.weight_2 = 0
        self.weight_s = None
    
    
    def __str__(self):
        return f"K1: {self.key_nodes_1}\nE1: {self.edges_1}\nK2: {self.key_nodes_2}\nE2: {self.edges_2}\nObj: {self.weight_1} + {self.weight_2} + {self.weight_s} = {self.evaluate()}"
        
    
    def evaluate(self):
        return self.weight_1 + self.weight_2 + self.weight_s


    # NOTE: this does NOT create a deep copy!
    def copy(self, t1 : bool, t2 : bool):
        s = Solution()

        if t1:
            s.key_nodes_1 = self.key_nodes_1
            s.edges_1 = self.edges_1
            s.weight_1 = self.weight_1
        
        if t2:
            s.key_nodes_2 = self.key_nodes_2
            s.edges_2 = self.edges_2
            s.weight_2 = self.weight_2
        
        if t1 and t2:
            s.weight_s = self.weight_s
        
        return s
    
    def write_solution_to_file(self, g : Graph, file : str):
        f = open(file, "w")
        
        f.write(f"S {g.abs_V} {g.abs_E} {g.abs_T_1} {g.abs_T_2}\n")
        
        for n1, n2 in self.edges_1:
            f.write(f"S1 {n1} {n2}\n")
            
        for n1, n2 in self.edges_2:
            f.write(f"S2 {n1} {n2}\n")
        
        f.close()