#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import operator
Item = namedtuple("Item", ['index', 'value', 'weight','density'])
Node = namedtuple("Node", ['level','elements', 'bound', 'weight', 'profit'])

class Node:
    def __init__(self, level, elements, bound, weight, profit):
        self.level = level
        self.elements = elements
        self.bound = bound
        self.weight = weight
        self.profit = profit
    
    def printNode(self):
        print("level: " + str(self.level) + ",elements: " + str(self.elements) + ",bound: " + str(self.bound)
              + ",weight: " + str(self.weight)+ ",profit: " + str(self.profit))

import time

class knapsack:
    def __init__(self, items):
        self.items = items
        
    def maxTuple(self, newVal, tuple1,tuple2):
        if tuple1[0] + newVal>tuple2[0]:
            return (tuple1[0] + newVal,tuple1[1])
        else:
            return tuple2   
    
    def runDynamicProgramming(self,weight,index, taken):
        if index == 0  or weight ==0: 
            return 0, taken
        item = self.items[index-1]
        if (item.weight > weight): 
            return self.runDynamicProgramming(weight, index -1,taken)       
        else: 
            n_taken = taken.copy()
            n_taken[index-1] = 1
            return self.maxTuple(item.value, self.runDynamicProgramming(weight-item.weight, index-1,n_taken), 
                       self.runDynamicProgramming(weight , index-1, taken)) 
            
    def bound(self, u, n, weight):
        if u.weight >= weight:
            return 0
        
        profit_bound = u.profit
        
        j = u.level + 1
        totweight = u.weight
        
        while ((j<n) and (totweight + self.items[j].weight <= weight)):
            totweight += self.items[j].weight
            profit_bound += self.items[j].value
            j+=1
            
        if (j<n):
            profit_bound += (weight-totweight)*self.items[j].value/self.items[j].weight
                
        return profit_bound
            
    def runBranchBound(self,weight,index, taken):
        maxProfit = 0
        queue = []
        u= Node(-1,[],0,0,0)
        v= Node(-1,[],0,0,0)
        queue.append(u) #Dummy node
        #print("Executing")
        #print(weight)
        while queue:
            #time.sleep(5)
            #print("Round")
 
            u = queue.pop(0)
            if u.level == -1:
                v.level = 0
            if u.level == len(self.items) -1:
                continue
            
            
            v.level = u.level + 1
            v.weight = u.weight + self.items[v.level].weight
            v.profit = u.profit + self.items[v.level].value
            
            #print("1")
            #v.printNode()
            #u.printNode()
            
            if v.weight<weight and v.profit>maxProfit:
                maxProfit = v.profit
            
            v.bound = self.bound(v,len(self.items),weight)
            
           # print("maxProfit: " + str(maxProfit))
            
            #print("2")
            #v.printNode()
            #u.printNode()
            
            if v.bound>maxProfit:
                queue.append(v)
                
            currentlevel = v.level     
            
            v= Node(-1,[],0,0,0)
            v.level = currentlevel
            v.weight = u.weight
            v.profit = u.profit
            v.bound = self.bound(v,len(self.items),weight)
            
            if v.bound>maxProfit:
                queue.append(v)
                
            v= Node(-1,[],0,0,0)
                
           
        print("Finish")
        return maxProfit
    
    def bound2(self, weight, start):
        profit_bound = 0
        while start<len(self.items):
            item = self.items[start]
            if weight>=item.weight:
                profit_bound+=item.value
                weight -= item.weight
            else:
                profit_bound += item.value * weight /item.weight
                break
            start+=1
            
        return profit_bound
            
   
    def runBranchBound2(self,weight):
        maxProfit = 0
        max_taken = [0]*len(self.items)
        queue = []
        bound = self.bound2(weight, 0)
        u= Node(0,[0]*len(self.items),bound,weight,0)
        queue.append(u) #Dummy node
        #print("Executing")
        #print(weight)
        start = time.time()
        time_going = 0
        
        
        while queue and time_going<900:
            #time.sleep(5)
            #print("Round")
 
            u = queue.pop()
            if u.weight<0:
                continue
            if u.bound<=maxProfit:
                continue
            if maxProfit < u.profit:
                    maxProfit = u.profit
                    max_taken = u.elements
                
            if u.level>= len(self.items):
                continue
            
            item = self.items[u.level]
            
            ## Don't use it
            
            notake_bound = self.bound2(u.weight, u.level +1)
            queue.append(Node(u.level +1 , u.elements , u.profit + notake_bound ,  u.weight , u.profit))
              
            ## Use it
            take_weight = u.weight - item.weight
            take_profit = u.profit + item.value
            take_bound = self.bound2(take_weight, u.level +1)
            take_taken = u.elements.copy()
            take_taken[u.level] = 1
            queue.append(Node(u.level +1 , take_taken , take_profit + take_bound ,  take_weight , take_profit))
            time_going = time.time() - start

           
        print("Finish")
        return (maxProfit , max_taken)         

    def grunGreedy(self,capacity):            
        value = 0
        weight = 0
        taken = [0]*len(self.items)
        
        for item in reversed(self.items):
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
                
        return value,taken
            
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1]),int(parts[0])/int(parts[1])))

#    model = knapsack(items)        
#    tuple_res = model.runBranchBound(capacity,item_count,[0]*len(items))

    # Sort algorithm
    items.sort(key = operator.itemgetter(3), reverse=True) 
    
    model_sort = knapsack(items)        
    if item_count>100000:
        tuple_res_sort = model_sort.grunGreedy(capacity)
        taken = tuple_res_sort[1]
    else:
        opt_bb = model_sort.runBranchBound2(capacity)

            
        opt_greedy = model_sort.grunGreedy(capacity)
        
        if opt_bb[0] > opt_greedy[0]:
            tuple_res_sort = opt_bb
            taken = [0]*len(items)
            for i in range(len(items)):
                taken[items[i].index] = opt_bb[1][i]
        else:
            tuple_res_sort = opt_greedy
            taken = tuple_res_sort[1]

    

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
#    value = 0
#    weight = 0
#    taken = [0]*len(items)
#

#            

    
#    value = 0
#    weight = 0
#    
#    for i in range(len(items)):
#        
#        value += items[i].value*tuple_res[1][i]
#        weight += items[i].weight*tuple_res[1][i]
    
    # prepare the solution in the specified output format
    #output_data = str(tuple_res[0]) + ' ' + str(0) + '\n'
    #output_data += ' '.join(map(str, taken_opt))
    #return output_data

    output_data = str(tuple_res_sort[0]) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
    #if 2 > 1:
        file_location = sys.argv[1].strip()
        #file_location = 'ks_19_0'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


#python ./solver.py ./data/ks_82_0
        
     