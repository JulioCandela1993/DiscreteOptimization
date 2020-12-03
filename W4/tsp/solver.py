#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import random
import matplotlib.pyplot as plt
from datetime import datetime
import time

Point = namedtuple("Point", ['x', 'y'])

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

import time

class optimizator():
    def __init__(self, points,nodeCount):
        self.points = points
        self.nodeCount = nodeCount
        self.solution = [i for i in range(0, nodeCount)]
        self.appearances = [0 for i in range(0, nodeCount)]
        
    def minimumNeighbors(self):
        nodes = set(range(self.nodeCount))
        ini_node = nodes.pop()
        sol=[]
        sol.append(ini_node)
        dist_max = float('inf')
        while (len(nodes) > 1):
            for test_node in nodes:
                new_len = length(self.points[sol[-1]], self.points[test_node])
                if new_len < dist_max:
                    dist_max = new_len
                    best_node = test_node
            dist_max = float('inf')
            sol.append(best_node)
            nodes.remove(best_node)
        sol.append(nodes.pop())
        return sol
            
    
    def run2_Opt(self,initial_sol, k, iterations,window):
        
        sol = initial_sol.copy()
        
        for i in range(0,iterations):

            while True:
                indexRandom_1 = random.randint(0,self.nodeCount-4)
                indexRandom_2 = random.randint(indexRandom_1,self.nodeCount-2)
                
                if self.appearances[indexRandom_1] <= i and self.appearances[indexRandom_1]<= i:
                    break

            node_1_in = sol[indexRandom_1]
            node_1_out = sol[indexRandom_1 + 1]
            
            node_2_in = sol[indexRandom_2]
            node_2_out = sol[indexRandom_2 + 1]
            
            #intersect_flag = intersect(self.points[node_1_in],self.points[node_1_out],
            #                               self.points[node_2_in],self.points[node_2_out])                        
            
            ori_len = length(self.points[node_1_in], self.points[node_1_out]) + length(self.points[node_2_in], self.points[node_2_out]) 
            new_len = length(self.points[node_1_in], self.points[node_2_in]) + length(self.points[node_1_out], self.points[node_2_out]) 
            
            if new_len<ori_len:
                sol[(indexRandom_1+1):(indexRandom_2+1)] = reversed(sol[(indexRandom_1+1):(indexRandom_2+1)])
                self.appearances[indexRandom_1] == i + window
                self.appearances[indexRandom_2] == i + window


            if random.random()<0.1:
                sol.append(sol.pop())
                self.appearances.append(self.appearances.pop())

        return sol
    
    def runk_Opt(self, initial_sol, nOpt, timelimit,per_window):
        
        new_sol = initial_sol.copy()
        
        start = time.time()
        
        firstPointElement = [[i,0] for i in range(self.nodeCount)]
        
        window = per_window * self.nodeCount
        some_opt = False
        for opt in range(nOpt):

            diff = time.time() - start
            if diff > timelimit:
                break;
            
            available = [i[0] for i in firstPointElement if i[1] <= opt]
            indexRandom_1 = random.randint(0,len(available)-1)
            node_1 = available[indexRandom_1]
            for index, item in enumerate(new_sol):
                if item == node_1:
                    indexRandom_1 = index
                    break

            firstPointElement[new_sol[indexRandom_1]][1] = opt + window
            
            new_sol = new_sol[indexRandom_1:] +  new_sol[:indexRandom_1] 
            indexRandom_1 = 0
            
            node_t1 = (indexRandom_1, new_sol[indexRandom_1])
            node_t2 = ((indexRandom_1 + 1)%self.nodeCount,new_sol[(indexRandom_1 + 1)%self.nodeCount])

            visited = []
            visited.append(node_t1[1])
            visited.append(node_t2[1])
            
            while True:
                #print(node_t2)
                dist_t1_t2 = length(self.points[node_t1[1]],self.points[node_t2[1]])
                found = False
                 
                rest_nodes = []
                rest_nodes.append(node_t2[1])
                rest_nodes.append(new_sol[(node_t2[0] +1)%self.nodeCount])
                
                for index,item in enumerate(new_sol):                    
                    if item not in visited and item not in rest_nodes:
                        dist_t2_t3 = length(self.points[node_t2[1]],self.points[new_sol[index]])
                        dist_t1_t4 = length(self.points[node_t1[1]],self.points[new_sol[index-1]])
                        dist_t3_t4 = length(self.points[new_sol[index]],self.points[new_sol[index-1]])
                        if dist_t1_t4 + dist_t2_t3 < dist_t1_t2 + dist_t3_t4:
                            found = True
                            node_t3 = (index ,item)
                            visited.append(node_t3[1])
                            break
                
                if found == True:
                    some_opt = True
                    previous_index = (node_t3[0]- 1)%self.nodeCount
                    node_t4 = (previous_index, new_sol[previous_index])
                    new_sol[ node_t2[0]: (node_t4[0] +1)] =  reversed(new_sol[node_t2[0]: (node_t4[0] +1)])                    
                    node_t2 = node_t4
                else:
                    break

        if some_opt == False:
            l = random.randint(2, len(new_sol) - 1)
            i = random.randint(0, len(new_sol) - l)
            new_sol[i:(i+l)] = reversed(new_sol[i:(i+l)])

        return new_sol
        
    def state_value(self,solution):
        obj = length(self.points[solution[-1]], self.points[solution[0]])
        for index in range(0, len(self.points)-1):
            obj += length(self.points[solution[index]], self.points[solution[index+1]])
        return obj    
        
    def two_opt(self, curr_sol):
    
        novel = curr_sol.copy()
        found = False
        # scan all edges
        '''
        for i in range(0, len(curr_sol)-2):
            for l in range(i+2, len(curr_sol)-1):
                if intersect(POINTS[curr_sol[i]], POINTS[curr_sol[i+1]],\
                             POINTS[curr_sol[l]], POINTS[curr_sol[l+1]]):
                    found = True
                    break
            if found: break
        '''
        # or try some random number of them
        limit = len(curr_sol)
        while limit > 0:
            i = random.randint(0, len(curr_sol)-4)
            l = random.randint(i+2, len(curr_sol)-2)
            if intersect(self.points[curr_sol[i]], self.points[curr_sol[i+1]],\
                         self.points[curr_sol[l]], self.points[curr_sol[l+1]]):
                found = True
                break
            limit -= 1
        if found:
            novel[i+1:l+1] = reversed(novel[i+1:l+1])
        else:
            l = random.randint(2, len(curr_sol) - 1)
            i = random.randint(0, len(curr_sol) - l)
            novel[i:(i+l)] = reversed(novel[i:(i+l)])
        return novel


    def local_search(self, guess, guess_val, time_limit=120):
        #tabu = [] # keep last visited states
        #tabu_size = 5000
        best = guess.copy()
        current = guess
        lost = 0
        counter = 0
        T = math.sqrt(len(self.points))
        alpha = 0.999
        min_T = 1e-8
        start = time.time()
        diff = time.time() - start
        while diff < time_limit:
            if T <= min_T:
                T = math.sqrt(len(self.points))
            #tabu.append(current)
            #neigh = self.find_neigh(current,[],counter)
            neigh = self.find_neigh_kopt(current)
            if neigh is not None:
                #tabu.append(neigh)
                #if len(tabu) == tabu_size + 1:
                    #tabu = tabu[1:]
                if self.accept(current, neigh, T):
                    current = neigh
                    if self.state_value(current) < self.state_value(best):
                        best = current
            else:
                lost += 1
            counter += 1
            T *= alpha
            diff = time.time() - start
        assert(self.assert_sol(best, len(self.points)))
        #print('Returning solution after {} iteration and {} lost iterations at {}'.format(counter, lost, datetime.now().time()))
        return best
   
    def find_neigh_kopt(self, current):
        return self.runk_Opt(current, 1 , 100000, 0.6)
    
    def find_neigh(self, current, tabu, counter):
        if counter == 0:
            rand = len(current) - 1
        else:
            rand = None
        neigh = self.two_opt(current)
        #if not neigh in tabu:
        #    return neigh
        #return None
        return neigh

    def accept(self,current, novel, temperature):
        old_val = self.state_value(current)
        new_val = self.state_value(novel)
        if new_val <= old_val:
            return True
        if math.exp(-abs(new_val - old_val) / temperature) > random.random():
            return True
        #if random.uniform(temperature, 1) < random.random():
           #return True
        return False
    

    
    def assert_sol(self,solution, tot):
        return len(set(solution)) == tot
    
    #############3
    ############# NEW SOLUTION HERE
    #############
    
    def runk_Opt_Penalty(self, initial_sol, cur_node, nOpt, timelimit,per_window, active_nodes, penalty, lambd):
        
        new_sol = initial_sol.copy()
        
        start = time.time()
        
        #firstPointElement = [[i,0] for i in range(self.nodeCount)]
        
        nodes_changed = []
        
        min_dist = 999999999999999999
        
        #window = per_window * self.nodeCount
        some_opt = False
        for opt in range(nOpt):

            diff = time.time() - start
            if diff > timelimit:
                break;
            
            #available = [i[0] for i in firstPointElement if i[1] <= opt]
            
            for idx, val in enumerate(new_sol):
                if val == cur_node : 
                    indexRandom_1 = idx
            
            #indexRandom_1 = cur_node#random.randint(0,len(available)-1)
            #node_1 = available[indexRandom_1]
            #for index, item in enumerate(new_sol):
            #    if item == node_1:
            #        indexRandom_1 = index
            #        break

            #firstPointElement[new_sol[indexRandom_1]][1] = opt + window
            
            new_sol = new_sol[indexRandom_1:] +  new_sol[:indexRandom_1] 
            indexRandom_1 = 0
                        
            node_t1 = (indexRandom_1, new_sol[indexRandom_1])
            node_t2 = ((indexRandom_1 + 1)%self.nodeCount,new_sol[(indexRandom_1 + 1)%self.nodeCount])

            visited = []
            visited.append(node_t1[1])
            visited.append(node_t2[1])
            
            found = False
            
            while True:
                #print(node_t2)
                dist_t1_t2 = length(self.points[node_t1[1]],self.points[node_t2[1]])
                
                 
                rest_nodes = []
                rest_nodes.append(node_t2[1])
                rest_nodes.append(new_sol[(node_t2[0] +1)%self.nodeCount])
                
                for index,item in enumerate(new_sol):                    
                    if item not in visited and item not in rest_nodes:
                        
                        dist_t2_t3 = length(self.points[node_t2[1]],self.points[new_sol[index]])
                        dist_t1_t4 = length(self.points[node_t1[1]],self.points[new_sol[index-1]])
                        dist_t3_t4 = length(self.points[new_sol[index]],self.points[new_sol[index-1]])
                        
                        #Penalties:
                        p1_2 = penalty[node_t1[1]][node_t2[1]] * lambd
                        p2_3 = penalty[node_t2[1]][new_sol[index]] * lambd
                        p1_4 = penalty[node_t1[1]][new_sol[index-1]] * lambd
                        p3_4 = penalty[new_sol[index]][new_sol[index-1]] * lambd
                        
                        new_candidate = dist_t1_t4 +p1_4 + dist_t2_t3 + p2_3
                        old_candidate = dist_t1_t2 + p1_2 + dist_t3_t4 + p3_4
                                                                        
                        if new_candidate < old_candidate  and new_candidate < min_dist:
                            
                            min_dist = dist_t1_t4 +p1_4 + dist_t2_t3 + p2_3                            
                            found = True
                            node_t3 = (index ,item)
                            visited.append(node_t3[1])
                            #break
                            
                break
                
            if found == True:
                some_opt = True
                previous_index = (node_t3[0]- 1)%self.nodeCount
                node_t4 = (previous_index, new_sol[previous_index])
                new_sol[ node_t2[0]: (node_t4[0] +1)] =  reversed(new_sol[node_t2[0]: (node_t4[0] +1)])                    
                
                nodes_changed.append(node_t1[1])
                nodes_changed.append(node_t2[1])
                nodes_changed.append(node_t3[1])
                nodes_changed.append(node_t4[1])
                break

        if some_opt == False:
            active_nodes[cur_node] = 0
            
            #l = random.randint(2, len(new_sol) - 1)
            #i = random.randint(0, len(new_sol) - l)
            #new_sol[i:(i+l)] = reversed(new_sol[i:(i+l)])
            
        #Return to original position
        #max_len = self.nodeCount - 1
        #new_sol = new_sol[-cur_node:] +  new_sol[:-cur_node] 

        return new_sol, active_nodes, penalty, nodes_changed
    
    def state_value_penalized(self,solution, penalty, lambd):
        
        node_in = solution[-1]
        node_out = solution[0]
        p = penalty[node_in][node_out]
        obj = length(self.points[solution[-1]], self.points[solution[0]]) + p * lambd
        
        
        for index in range(0, len(self.points)-1):
            node_in = solution[index]
            node_out = solution[index+1]
            p = penalty[node_in][node_out]
            obj += length(self.points[node_in], self.points[node_out]) + p * lambd
        return obj  
    
    def manage_Penalties(self,active_nodes, penalty, cur_sol):
        
        max_dist = 0
        max_node = -1
        max_next_node = -1
        
        for index,item in enumerate(cur_sol):
            next_index = (index +1)%len(cur_sol)
            next_item = cur_sol[next_index]
            
            dist = length(self.points[item], self.points[next_item])
            p = 1 + penalty[item][next_item]
            
            factor = (dist/(1+p))
            
            if factor> max_dist:
                max_dist = factor
                max_node = item
                max_next_node = next_item
                
        penalty[max_node][max_next_node] +=1
        penalty[max_next_node][max_node] +=1
        
        active_nodes[max_node]   = 1 
        active_nodes[max_next_node]   = 1 
            
        return active_nodes, penalty
    
    def tsp_opt(self, ini_sol, time_limit):
        #Control by time! Less than 500
        start = time.time()
        diff = time.time() - start
        
        penalty = [[0 for x in range(len(ini_sol))] for y in range(len(ini_sol))] 
        
                ## Parameters
        alpha = 0.1
        lambd = 0
        
        cur_sol = ini_sol.copy()
        cur_sol_dist = self.state_value(cur_sol)
        cur_sol_dist_pen = self.state_value_penalized(cur_sol, penalty, lambd)
        
        best_sol_dist = cur_sol_dist
        best_sol = cur_sol
        
        active_nodes = [1 for x in range(len(ini_sol))] 
        
        while diff < time_limit:
            active_count =  len(list(filter(lambda number: number == 1, active_nodes) )  )    
            while active_count>0 and diff < time_limit:
                   
                ### Iteration
                for cur_node in  range(len(active_nodes)):
                    
                    if active_nodes[cur_node] == 0:
                        continue

                    
                    #print(active_nodes)
                    #print(cur_sol)
                    cur_sol, active_nodes, penalty, nodes_changed = self.runk_Opt_Penalty(cur_sol, cur_node, 1 , 100000, 1, active_nodes, penalty, lambd)
                    #print(active_nodes)
                    #print(cur_sol)
                    #print(nodes_changed)
                    cur_sol_dist = self.state_value(cur_sol)
                    
                    for i in nodes_changed:
                        active_nodes[i] = 1
                    
                    if cur_sol_dist < best_sol_dist:
                        best_sol_dist = cur_sol_dist
                        best_sol = cur_sol.copy()
                    
                active_count =  len(list(filter(lambda number: number == 1, active_nodes) )  )
                
                diff = time.time() - start
            
            if lambd == 0:
                lambd = alpha * self.state_value(ini_sol)/ len(ini_sol)
                
            active_nodes, penalty = self.manage_Penalties(active_nodes, penalty, cur_sol)
            
            
        
        print("Done")
        return best_sol

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    #solution = range(0, nodeCount
    optimizer = optimizator(points,nodeCount)
    ini_sol = optimizer.minimumNeighbors()
    #solution = optimizer.tsp_opt(ini_sol, 100)
    
    if nodeCount>=1500:
        solution = optimizer.tsp_opt(ini_sol, 5000)
    elif nodeCount>=500:
        solution = optimizer.tsp_opt(ini_sol, 1000)
    else:
        solution = optimizer.tsp_opt(ini_sol, 100)
    
#    for i in range(2):
#        if nodeCount>=1000:
#            time_limit = 2000
#        else:
#            time_limit = 1000
#        sol = optimizer.local_search(ini_sol, optimizer.state_value(ini_sol), time_limit)
#        if optimizer.state_value(sol) < optimizer.state_value(solution):
#            solution = sol
#        ini_sol = optimizer.runk_Opt(ini_sol,100000000,10,0.6)
    
#    k2sol = optimizer.run2_Opt(ini_sol,2,100000 ,10)
#    solution1 = optimizer.runk_Opt(ini_sol,1000000000000000,600,0.6)
#    solution2 = optimizer.runk_Opt(ini_sol,1000000000000000,600,0.6)
#    solution3 = optimizer.runk_Opt(ini_sol,1000000000000000,600,0.6)
#    
#    res1= optimizer.state_value(solution1)
#    res2= optimizer.state_value(solution2)
#    res3= optimizer.state_value(solution3)
#    res4= optimizer.state_value(k2sol)
#    res = min(res1,res2,res3,res4)
#    if res == res1:
#        solution = solution1
#    elif res == res2:
#        solution = solution2
#    elif res == res3:
#        solution = solution3
#    else:
#        solution = solution4
        
        
        
        
    #solution = optimizer.runk_Opt(ini_sol, 1)
#    fig, ax = plt.subplots()
#
#    ax.scatter([i.x for i in points], [i.y for i in points])
#
#    for i, txt in enumerate(solution):
#        ax.annotate(str(i) + "-" + str(txt), (points[txt].x, points[txt].y))
#
#    plt.show()

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys
import os

#os.chdir('G:\\Documentos\\Learning\\Coursera\\DiscreteOptimization\\W4\\tsp')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

