#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from mip import Model, xsum, minimize, BINARY
from copy import deepcopy
import random
import time
from datetime import datetime

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

class optimizator():
    def __init__(self, vehicle_count,vehicle_capacity,customers,depot):
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.customers = customers
        self.depot = depot


    def run_mip(self):
        m = Model("Vehicle Routing")

        num_customers = set(range(len(self.customers)))
        num_vehicles = set(range(self.vehicle_count))
        
        ## Sequence of deliveries by vehicles
        sequence_vehicles = [[[m.add_var( name = "V" + str(i) + "S" + str(j) + "D" + str(k) , var_type=BINARY) 
                                for k in num_customers] for j in num_customers] for i in num_vehicles]
        
        # SubTour Elimination
        sub_tour = [[m.add_var(name = "V" + str(i) + "ST" + str(i) )  for j in num_customers] for i in num_vehicles]
        
        
        #Minimize the distance from vehicle from point to point
        m.objective = minimize(xsum( length(self.customers[j],self.customers[k])*sequence_vehicles[i][j][k] 
                                            for i in num_vehicles 
                                            for j in num_customers 
                                            for k in num_customers))
                                
        
        for i in num_vehicles:
            for j in num_customers:
                for k in num_customers:
                    m+= sequence_vehicles[i][j][k] + sequence_vehicles[i][k][j] <= 1
                                         
        for j in num_customers:
            if j>0:
                m += xsum(sequence_vehicles[i][j][k] for i in num_vehicles for k in num_customers) == 1
                
        # Each vehicle can only visit once a place
        for k in num_customers:
            if k>0:
                m += xsum(sequence_vehicles[i][j][k] for i in num_vehicles for j in num_customers) == 1
         
        # Circuit    
        for i in num_vehicles:
            for j in num_customers:
                m += xsum(sequence_vehicles[i][j][k] for k in num_customers) - xsum(sequence_vehicles[i][k][j] for k in num_customers) == 0    
            
        # Start and finish in the same position (depot 0)
        for i in num_vehicles:
                m += xsum(sequence_vehicles[i][0][k] for k in num_customers) == xsum(sequence_vehicles[i][k][0] for k in num_customers) 

        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) <= 1
                
        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) >= xsum(sequence_vehicles[i][j][k] for k in num_customers for j in num_customers)/len(num_customers)            

        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) <= xsum(sequence_vehicles[i][j][k] for k in num_customers for j in num_customers)                   
                
        for j in num_customers:
            if j>0:
                m += xsum(sequence_vehicles[i][j][k] for k in num_customers for i in num_vehicles) + xsum(sequence_vehicles[i][k][j] for k in num_customers for i in num_vehicles) == 2    
            
        # Capacity of vehicles less than the demand.        
        for i in num_vehicles:
            m += xsum( sequence_vehicles[i][j][k] * self.customers[j].demand  for j in num_customers for k in num_customers) <= self.vehicle_capacity

        
        m.optimize(max_seconds=1000)
        
        assignation = m.vars
        
        solution = [[(j,k) for j in num_customers for k in num_customers
                     if assignation[i*len(num_customers)*len(num_customers) + j *len(num_customers) + k ].x>0.99] 
                                for i in num_vehicles] 

        print(solution)        
        #selectedCustomers = [ i for j in num_customers for i in num_facilities] if assignations[j*i + i].x>=0.99]
        #len(selectedCustomers)
        return (m.objective_value, solution)

    def run_mip_2(self):
        m = Model("Vehicle Routing")

        num_customers = set(range(len(self.customers)))
        num_vehicles = set(range(self.vehicle_count))
        
        ## Sequence of deliveries by vehicles
        sequence_vehicles = [[[m.add_var( name = "S" + str(i) + "D" + str(j) + "V" + str(k) , var_type=BINARY) 
                                for k in num_vehicles] for j in num_customers] for i in num_customers]
        
        #SubTourElimination
        
        #Minimize the distance from vehicle from point to point
        m.objective = minimize(xsum( length(self.customers[i],self.customers[j])*sequence_vehicles[i][j][k] 
                                            for i in num_customers 
                                            for j in num_customers 
                                            for k in num_vehicles))

        for i in num_customers:
                m += xsum( sequence_vehicles[i][j][k]  for j in num_customers for k in num_vehicles) == 1
                
        for k in num_vehicles:
                m += xsum( sequence_vehicles[0][j][k]  for j in num_customers if j>0 ) <= 1
                
        for k in num_vehicles:
            for i in num_customers :
                m += xsum( sequence_vehicles[i][j][k]  for j in num_customers) - xsum( sequence_vehicles[j][i][k]  for j in num_customers) == 0
            
        
        for k in num_vehicles:
                m += xsum(sequence_vehicles[0][j][k] for j in num_customers) == xsum(sequence_vehicles[j][0][k] for j in num_customers) 
             
        # Capacity of vehicles less than the demand.        
#        for i in num_vehicles:
#            m += xsum( sequence_vehicles[i][j][k] * self.customers[j].demand  for j in num_customers for k in num_customers) <= self.vehicle_capacity

        
        m.optimize(max_seconds=1000)
        
        assignation = m.vars
        
        solution = [[(j,k) for i in num_customers for j in num_customers
                     if assignation[i*len(num_customers)*len(num_vehicles) + j *len(num_vehicles) + k ].x>0.99] 
                                for k in num_vehicles] 

        print(solution)        
        #selectedCustomers = [ i for j in num_customers for i in num_facilities] if assignations[j*i + i].x>=0.99]
        #len(selectedCustomers)
        return (m.objective_value, solution)


    def run_mip_3(self, time):
        m = Model("Vehicle Routing")

        num_customers = set(range(len(self.customers)))
        num_vehicles = set(range(self.vehicle_count))
        
        ## Sequence of deliveries by vehicles
        sequence_vehicles = [[[m.add_var( name = "V" + str(i) + "S" + str(j) + "D" + str(k) , var_type=BINARY) 
                                for k in num_customers] for j in num_customers] for i in num_vehicles]
        
        # SubTour Elimination
        sub_tour = [[m.add_var(name = "V" + str(i) + "ST" + str(i) )  for j in num_customers] for i in num_vehicles]
        
        
        #Minimize the distance from vehicle from point to point
        m.objective = minimize(xsum( length(self.customers[j],self.customers[k])*sequence_vehicles[i][j][k] 
                                            for i in num_vehicles 
                                            for j in num_customers 
                                            for k in num_customers))
                                
        
        for i in num_vehicles:
            for j in num_customers:
                for k in num_customers:
                    m+= sequence_vehicles[i][j][k] + sequence_vehicles[i][k][j] <= 1
                                         
        for j in num_customers:
            if j>0:
                m += xsum(sequence_vehicles[i][j][k] for i in num_vehicles for k in num_customers) == 1
                
        # Each vehicle can only visit once a place
        for k in num_customers:
            if k>0:
                m += xsum(sequence_vehicles[i][j][k] for i in num_vehicles for j in num_customers) == 1
                
        for i in num_vehicles:
            for j in num_customers:
                for k in num_customers:
                    if j != k and j>0 and k>0:
                        m += sub_tour[i][j] - (len(num_customers) +1 ) * sequence_vehicles[i][j][k] >= sub_tour[i][k] - len(num_customers)
                        
         
        # Circuit    
        for i in num_vehicles:
            for j in num_customers:
                m += xsum(sequence_vehicles[i][j][k] for k in num_customers) - xsum(sequence_vehicles[i][k][j] for k in num_customers) == 0    
            
        # Start and finish in the same position (depot 0)
        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) <= 1
                
        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) >= xsum(sequence_vehicles[i][j][k] for k in num_customers for j in num_customers)/len(num_customers)            

        for i in num_vehicles:
            m += xsum(sequence_vehicles[i][0][k] for k in num_customers) <= xsum(sequence_vehicles[i][j][k] for k in num_customers for j in num_customers)                   
                
        for j in num_customers:
            if j>0:
                m += xsum(sequence_vehicles[i][j][k] for k in num_customers for i in num_vehicles) + xsum(sequence_vehicles[i][k][j] for k in num_customers for i in num_vehicles) == 2    
            
        # Capacity of vehicles less than the demand.        
        for i in num_vehicles:
            m += xsum( sequence_vehicles[i][j][k] * self.customers[j].demand  for j in num_customers for k in num_customers) <= self.vehicle_capacity

        
        m.optimize(max_seconds=time)
        
        assignation = m.vars
        
        solution = [[(j,k) for j in num_customers for k in num_customers
                     if assignation[i*len(num_customers)*len(num_customers) + j *len(num_customers) + k ].x>0.99] 
                                for i in num_vehicles] 

        print(solution)        
        #selectedCustomers = [ i for j in num_customers for i in num_facilities] if assignations[j*i + i].x>=0.99]
        #len(selectedCustomers)
        return (m.objective_value, solution)
    
    def trivial_sol(self,customers, depot, vehicle_count, vehicle_capacity):
        vehicle_tours = []
        
        remaining_customers = set(customers)
        remaining_customers.remove(depot)
        
        for v in range(0, vehicle_count):
            # print "Start Vehicle: ",v
            vehicle_tours.append([])
            capacity_remaining = vehicle_capacity
            while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
                used = set()
                order = sorted(remaining_customers, key=lambda customer: -customer.demand)
                for customer in order:
                    if capacity_remaining >= customer.demand:
                        capacity_remaining -= customer.demand
                        vehicle_tours[v].append(customer)
                        # print '   add', ci, capacity_remaining
                        used.add(customer)
                remaining_customers -= used
        return vehicle_tours  
    
    def state_value(self,veh_tours):
        obj = 0
        for v in range(len(veh_tours)):
            vehicle_tour = veh_tours[v]
            if len(vehicle_tour) > 0:
                obj += length(self.depot, vehicle_tour[0])
                for i in range(0, len(vehicle_tour)-1):
                    obj += length(vehicle_tour[i], vehicle_tour[i+1])
                obj += length(vehicle_tour[-1], self.depot)    
        return obj
    
    def find_neigh(self,curr_sol, customers, vehicle_capacity):
        neigh = deepcopy(curr_sol)
        r1 = random.randint(1, len(customers)-1) # what customer to move...
        d = customers[r1].demand
        available = []
        for veh_tour in neigh:
            if customers[r1] in veh_tour:
                veh_tour.remove(customers[r1])
            tot = vehicle_capacity
            for cus in veh_tour:
                tot -= cus.demand
            available.append(tot)
        s = [(i,x) for (i,x) in enumerate(available) if x>=d]
        r2 = s[random.randint(0, len(s)-1)][0] # ... to what vehicle...
        r3 = random.randint(0, len(curr_sol[r2])) # ... in what position
        neigh[r2].insert(r3, customers[r1])
        return neigh
    
    def local_search(self,customers, guess, vehicle_capacity, time_limit=120):
        best = deepcopy(guess)
        current = guess
        restart = 0
        counter = 0
        T = len(customers)
        alpha = 0.999
        min_T = 1e-8
        start = time.time()
        diff = time.time() - start
        print('Local search starts at {}'.format(datetime.now().time()))
        while diff < time_limit:
            if T <= min_T:
                T = len(customers)
                restart += 1
            if len(customers) < 95:
                neigh = self.find_neigh_2(current, customers, vehicle_capacity)
            else:
                neigh = self.find_neigh(current, customers, vehicle_capacity)
            if neigh is not None:
                if self.accept(current, neigh, T):
                    current = neigh
                    if self.state_value(current) < self.state_value(best):
                        best = deepcopy(current)
            counter += 1
            T *= alpha
            diff = time.time() - start
        print('Returning solution after {} iteration and {} restarts at {}'.format(counter, restart, datetime.now().time()))
        return best
    
    def accept(self,current, novel, temperature):
        old_val = self.state_value(current)
        new_val = self.state_value(novel)
        if new_val <= old_val:
            return True
        if math.exp(-abs(new_val - old_val) / temperature) > random.random():
            return True
        return False
    
    def find_neigh_2(self,curr_sol, customers, vehicle_capacity):
        neigh = deepcopy(curr_sol)
        v1 = random.randint(0, len(curr_sol)-1) # from what vehicle
        while len(curr_sol[v1]) == 0:
            v1 = random.randint(0, len(curr_sol)-1)
        c1 = random.randint(0, len(curr_sol[v1])-1) # what customer
        tmp = neigh[v1][c1]
        cap1, cap2 = vehicle_capacity+1, 0
        while cap1 > vehicle_capacity or cap2 < tmp.demand:
            v2 = random.randint(0, len(curr_sol)-1) # to what vehicle
            c2 = random.randint(0, len(curr_sol[v2])) # in what position
            cap1 = sum(curr_sol[v1][x].demand for x in range(len(curr_sol[v1])) if x!=c1)
            if c2 < len(curr_sol[v2]):
                 cap1 += curr_sol[v2][c2].demand
            cap2 = vehicle_capacity - sum(curr_sol[v2][x].demand for x in range(len(curr_sol[v2])) if x!=c2)
        if c2 < len(curr_sol[v2]):
            neigh[v1][c1] = neigh[v2][c2]
            neigh[v2][c2] = tmp
        else:
            neigh[v1].remove(tmp)
            neigh[v2].insert(c2, tmp)
        return neigh


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 

    vehicle_tours = []

    optimizer = optimizator(vehicle_count,vehicle_capacity,customers, depot)
    
    if customer_count <= 70:
        if customer_count <=20:
            sol = optimizer.run_mip_3(100)
        elif customer_count <=50:
            sol = optimizer.run_mip_3(1000)
        else:
            sol = optimizer.run_mip_3(1500)
        route = sol[1]
    
        for i in range(0, vehicle_count):
            int_list = [0]
            tour_i = route[i]
            first = True
            if len(tour_i)>0:           
                while True:
                    for source_dest in tour_i:
                        if first :
                            select_d = source_dest[1]
                            first = False
                            int_list.append(select_d)
                            break
                        else:
                            if source_dest[0] == select_d:
                                select_d = source_dest[1]
                                if select_d == 0:
                                    break
                                int_list.append(select_d)
                                break
                    if select_d == 0:
                        break
            int_list.append(0)
            vehicle_tours.append(int_list)
            vehicle_tours[0]
        obj = sol[0]
        
    else:
        vehicle_tours_trivial  = optimizer.trivial_sol(customers, depot, vehicle_count, vehicle_capacity)
        vehicle_tours_local = optimizer.local_search(customers, vehicle_tours_trivial, vehicle_capacity, 1000)
        
        for veh in vehicle_tours_local:
            int_list = [0]
            for cus in veh:
                int_list.append(cus.index)
            int_list.append(0)
            vehicle_tours.append(int_list)      
        obj = optimizer.state_value(vehicle_tours_local)
    
    
    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(customer) for customer in vehicle_tours[v]])  + '\n'

    return outputData


import sys
#import os
#os.chdir("G:\\Documentos\\Learning\\Coursera\\DiscreteOptimization\\W7\\vrp\\data")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
    #if True:
        file_location = sys.argv[1].strip()
        #file_location = "vrp_5_4_1"
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

