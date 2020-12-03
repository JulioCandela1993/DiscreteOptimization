#!/usr/bin/python
# -*- coding: utf-8 -*-

import networkx as nx
from psutil import cpu_count
import random

class Tabu():
    def __init__(self, tabu_size):
        self.tabu_size = tabu_size
        self.tabu_queue = []
        
    def tabuappend(self,element):
        if len(self.tabu_queue) >= self.tabu_size:
            self.tabu_queue.pop(0)
        self.tabu_queue.append(element)
            

class optimizator():
    def __init__(self, node_count, edges):
        self.node_count = node_count
        self.edges = edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(node_count))
        self.graph.add_edges_from(edges)
        
    def run(self,time_limit):
        problem = Model("graph_coloring")
        problem.setParam('OutputFlag', False)
        problem.setParam("TimeLimit", time_limit)
        problem.setParam("Threads", cpu_count())
        first_guess_count, opt, first_guess = self.greedy()
        
        ### Less Color than greedy
        colors = problem.addVars(first_guess_count, vtype=GRB.BINARY , name="bincolor")
        nodes = problem.addVars(self.node_count, first_guess_count, vtype=GRB.BINARY, name="nodecolor")
        
        for i in range(first_guess_count):
            colors[i].setAttr("Start", 0)
            for j in range(self.node_count):
                nodes[(j, i)].setAttr("Start", 0)
        
        for i, j in enumerate(first_guess):
            colors[j].setAttr("Start", 1)
            nodes[(i, j)].setAttr("Start", 1)
                
        problem.setObjective(quicksum(colors), GRB.MINIMIZE)
        
        #Only one color
        problem.addConstrs((nodes.sum(i, "*") == 1
                  for i in range(self.node_count)),
                 name="1_color")
    
        #colors assigned
        problem.addConstrs((nodes[(i, k)] - colors[k] <= 0
                  for i in range(self.node_count)
                  for k in range(first_guess_count)),
                 name="colors_assigned")
    
        #different colors
        problem.addConstrs((nodes[(edge[0], k)] + nodes[(edge[1], k)] <= 1
                  for edge in self.edges
                  for k in range(first_guess_count)),
                 name="different")
    
        # color index should be as low as possible
        problem.addConstrs((colors[i-1] - colors[i] >= 0
                      for i in range(1,first_guess_count)),
                     name="low index")
    
        # initialize with 0
        problem.addConstrs((nodes[(i,i)] == 1 for i in range(1)),
                     name="base color")

        problem.update()
        problem.optimize()
        
        bin_best_sol = [int(var.x) for var in problem.getVars()]
        best_sol_num = sum(bin_best_sol[:first_guess_count])
        best_sol_array = bin_best_sol[first_guess_count:]
        best_sol = [j 
                                for i in range(self.node_count) 
                                for j in range(first_guess_count)
                                if best_sol_array[i*first_guess_count + j] == 1]
        
        return best_sol_num, best_sol
        
    def greedy(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(self.node_count))
        graph.add_edges_from(self.edges)
    
        strategies = [nx.coloring.strategy_largest_first,
                      nx.coloring.strategy_random_sequential,
                      nx.coloring.strategy_smallest_last,
                      nx.coloring.strategy_independent_set,
                      nx.coloring.strategy_connected_sequential_bfs,
                      nx.coloring.strategy_connected_sequential_dfs,
                      nx.coloring.strategy_connected_sequential,
                      nx.coloring.strategy_saturation_largest_first]
    
        best_color_count, best_coloring = self.node_count, {i: i for i in range(self.node_count)}
        for strategy in strategies:
            curr_coloring = nx.coloring.greedy_color(G=graph, strategy=strategy)
            curr_color_count = max(curr_coloring.values()) + 1
            if curr_color_count < best_color_count:
                best_color_count = curr_color_count
                best_coloring = curr_coloring
        return best_color_count, 0, [best_coloring[i] for i in range(self.node_count)]

    def runLocalSearch(self):
        sol = [0 for i in range(self.node_count)]
        priority = [0 for i in range(self.node_count)]
        edge_aux = []
        node_priority = 0
        value_priority = 0
        for edge in self.edges:
            if edge[0] > edge[1]:
                edge_aux.append([edge[1],edge[0]])
            if sol[edge[0]] == sol[edge[1]]:
                priority[edge[0]]+=1
                priority[edge[1]]+=1
                if priority[edge[0]]>value_priority:
                    node_priority = edge[0]
                    value_priority=priority[edge[0]]
                if priority[edge[1]]>node_priority:
                    node_priority = edge[1]
                    value_priority=priority[edge[1]]
                edge_aux.append(edge)
        while(True):   
            for edge in edge_aux:
                if edge[0] == node_priority or edge[1] == node_priority:
                    e_1 = node_priority
                    if edge[0] == node_priority: 
                        e_2 = edge[1]
                    else: 
                        e_2 = edge[0]
                    if sol[e_1] == sol[e_2]:
                        sol[e_2] +=1
                        constraint=True
            ## Update Priority Constraints
            priority = [0 for i in range(self.node_count)]
            node_priority = 0
            value_priority = 0
            for edge in edge_aux:
                if sol[edge[0]] == sol[edge[1]]:
                    priority[edge[0]]+=1
                    priority[edge[1]]+=1
                    if priority[edge[0]]>value_priority:
                        node_priority = edge[0]
                        value_priority=priority[edge[0]]
                    if priority[edge[1]]>value_priority:
                        node_priority = edge[1]
                        value_priority=priority[edge[1]]
                
            if constraint == False:
                break
            constraint = False
        return len(set(sol)),sol
    
      
    def runExperiment(self):
        
        sol = [0 for i in range(self.node_count)]
        edge_aux = []
        for edge in self.edges:
            if edge[0] > edge[1]:
                edge_aux.append([edge[1],edge[0]])
            else:
                edge_aux.append(edge)
        
        while(True):        
            for edge in edge_aux:
                if sol[edge[0]] == sol[edge[1]]:
                    sol[edge[1]] +=1
                    constraint = True
                    break
            if constraint == False:
                break
            constraint = False
        return len(set(sol)),sol
    
    def TabuSearch(self):
        
        best_sol_num, opt, best_sol = self.greedy()
        
        ini_num_colors = best_sol_num
        
        tabulimit = len(best_sol) / 10
        
        solution = []
        solution_count = -1
        for cur_color in reversed(range(ini_num_colors)):
            
            if cur_color == 0:
                return solution
            
            print("Iteration")
            print(cur_color)
            num_attempts = 100
            counter_attempts = 0
            
            while True:
                feasible = self.findBetterSolution(cur_color+1, tabulimit, best_sol)               
                if feasible:
                    solution = feasible
                    solution_count = cur_color+1
                    best_sol = self.remove_last_color(feasible)
                    print("Atempt: "+str(counter_attempts))
                    print(cur_color , solution)
                    print(cur_color , feasible)
                    print(cur_color , best_sol)
                    break;
                
                counter_attempts +=1
                
                if counter_attempts>= num_attempts:
                    
                    # No more possible optimization, so finish the search
                    return (solution_count, solution)
        
        return (solution_count, solution)
    
    def remove_last_color(self, best_sol):
        max_color = max(best_sol)
        return [max_color-1 if col == max_color and col>0 else col  for col in best_sol]
    
    def calculateViolations(self, sol):
        num_violations = 0
        vect_violation = []
        
        for i in range(self.node_count):
            node_violations = 0
            for n in self.graph.neighbors(i):
                node_violations += sol[i] == sol[n]
            vect_violation.append(node_violations)
            num_violations += node_violations

        return (num_violations, vect_violation)
    
    def findBetterSolution(self, color_count, tabulimit, best_sol):
        
        num_steps = 70000
        count_step = 0
        
        num_violations, vect_violation  = self.calculateViolations(best_sol)

        tabu = Tabu(tabulimit)
        
        while count_step <= num_steps and num_violations>0:
    
            node = self.select_warning_node(vect_violation, best_sol, tabu)
            while node == -1:
                tabu.tabu_queue.pop(0)
                node = self.select_warning_node(vect_violation, best_sol, tabu)
                
            tabu.tabuappend(node)
            
            best_sol, num_violations, vect_violation  = self.change_color(best_sol, node , num_violations, vect_violation, color_count)
            
            count_step += 1
            
            #print("Feasble: " + str(count_step))
            #print("Node: " + str(node))
            #print(best_sol)
            #print(num_violations)
            #print(vect_violation)
            
        if count_step >= num_steps:
            return []
        
        return best_sol
    
    def change_color(self, best_sol, node , num_violations, vect_violation, cur_color):
        
        color_vector = [0 for i in range(cur_color)]
        
        
        # Different colors of the neighbor
        for nb in self.graph.neighbors(node):
            color_neighbor = best_sol[nb]
            color_vector[color_neighbor] += 1
            
        min_color_viol = 9999999999999
        color_vector_viol =[]
        
        for col_trial in range(cur_color):
            if col_trial ==  best_sol[node]:
                continue
            if color_vector[col_trial] < min_color_viol:
                min_color_viol = color_vector[col_trial]
                color_vector_viol.clear()
                color_vector_viol.append(col_trial)
            elif color_vector[col_trial] == min_color_viol:
                color_vector_viol.append(col_trial)
                
        ## Send some warning
        
        new_color = random.choice(color_vector_viol)
        
        ## Update violation vector
        
        for nb in self.graph.neighbors(node):
            if best_sol[nb] == best_sol[node]:
                vect_violation[nb] -= 1
                vect_violation[node] -= 1
                num_violations -=2
            if best_sol[nb] == new_color:
                vect_violation[nb] += 1
                vect_violation[node] += 1
                num_violations +=2
        
        best_sol[node] = new_color
        
        return best_sol, num_violations, vect_violation
        
    
    def select_warning_node(self,vect_violation, best_sol, tabu):
        
        max_violation = 0
        max_violation_vector = []
        
        for node in range(len(vect_violation)):
            if vect_violation[node] == 0:
                continue
            if node in tabu.tabu_queue:
                continue

            if vect_violation[node]> max_violation:
                max_violation = vect_violation[node]
                max_violation_vector.clear()
                max_violation_vector.append(node)
            elif vect_violation[node] == max_violation:
                max_violation_vector.append(node)            
        
        if not max_violation_vector:
            return -1
        
        return random.choice(max_violation_vector)
              
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))


    
    if node_count<=0:
        
        solver = optimizator(node_count,edges)
        best_sol_num, best_sol = solver.runLocalSearch()
        
#        
    # build a trivial solution
    # every node has its own color

    else: 
        solver = optimizator(node_count,edges)
        best_sol_num, best_sol = solver.TabuSearch()
        #best_sol_num, opt, best_sol = solver.greedy()
    
    # prepare the solution in the specified output format
    output_data = str(best_sol_num) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_sol))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print(file_location)
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

