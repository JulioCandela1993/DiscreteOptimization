#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from mip import Model, xsum, minimize, BINARY
import math
import time
from ortools.linear_solver import pywraplp
import numpy as np

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


class optimizator():
    def __init__(self, facilities,customers):
        self.facilities = facilities
        self.customers = customers
        
        self.matrix_distances = [[ 0 for f in self.facilities] for c in self.customers]
        
        for c in self.customers:
            for f in self.facilities:
                self.matrix_distances[c.index][f.index] = length(c.location, f.location)
        
        


    def run_mip(self):
        m = Model("Facility_Warehouse")
        num_facilities = set(range(len(self.facilities)))
        num_customers = set(range(len(self.customers)))
        work_warehouse = [m.add_var(var_type=BINARY) for i in num_facilities]
        assign_customer = [[m.add_var(name = "C_" + str(j) + 'F_' + str(i), var_type=BINARY) for i in num_facilities] for j in num_customers]
        
        m.objective = minimize(xsum( (work_warehouse[i]*self.facilities[i].setup_cost)/len(num_customers) 
                        + length(self.customers[j].location , self.facilities[i].location) * assign_customer[j][i]                                                           
                                    for i in num_facilities for j in num_customers ))
      

        # constraint : customer only assigned to one DWH
        for i in num_customers:
            for j in num_facilities:
                m += assign_customer[i][j] <=  work_warehouse[j]

        # constraint : customer only assigned to one DWH
        for i in num_customers:
            m += xsum(assign_customer[i][j] for j in num_facilities) == 1
            
        # constraint : warehouse can allow its capacity
        for i in num_facilities:
            m += xsum(assign_customer[j][i]*self.customers[j].demand for j in num_customers) <= self.facilities[i].capacity
            
        m.optimize(max_seconds=1000)

        assignation = m.vars[len(num_facilities):]
        
        solution = [j for i in num_customers for j in num_facilities if assignation[i*len(num_facilities) + j].x>0.99]    
                
        #selectedCustomers = [ i for j in num_customers for i in num_facilities] if assignations[j*i + i].x>=0.99]
        #len(selectedCustomers)
        return (m.objective_value, solution)


    def initial_solution(self):
        
        solution = [-1]*len(self.customers)
        capacity_remaining = [f.capacity for f in self.facilities]
    
        facility_index = 0
        for customer in self.customers:
            if capacity_remaining[facility_index] >= customer.demand:
                solution[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution[customer.index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
    
        used = [0]*len(self.facilities)
        for facility_index in solution:
            used[facility_index] = 1
    
        # calculate the cost of the solution
        obj = sum([f.setup_cost*used[f.index] for f in self.facilities])
        for customer in self.customers:
            obj += length(customer.location, self.facilities[solution[customer.index]].location)
            
        return obj, solution
        
   
    def objective_function(self, solution):
        obj = 0
        used = [0 for i in self.facilities]
        for customer in self.customers:
            obj += length(customer.location, self.facilities[solution[customer.index]].location)
            used[solution[customer.index]] = 1
            
        obj += sum([f.setup_cost*used[f.index] for f in self.facilities])
        
        return obj
        
     
    def select_best_customer(self, cur_sol, facilities_clients, facilities_available, penalty,lambd):
        
        max_gain = 0
        choose_customer = -1
        choose_facility_old = -1
        choose_facility_new = -1
        
        for c in self.customers:
            facility_old = cur_sol[c.index]
            available_old = facilities_available[facility_old]
            customer = c.index
            for f in self.facilities:
                facility_new = f.index
                available_new = facilities_available[facility_new]
                if facility_old == facility_new:
                    continue
                
                if  available_new < c.demand:
                    continue
                
                cost_old = self.matrix_distances[customer][facility_old] + (1 if len(facilities_clients[facility_old]) == 1 else 0) * self.facilities[facility_old].setup_cost
                cost_new = self.matrix_distances[customer][facility_new] + (1 if len(facilities_clients[facility_new]) == 0 else 0) * self.facilities[facility_new].setup_cost        
                
                cost_old_pen = cost_old + lambd*penalty[customer][facility_old]
                cost_new_pen = cost_new + lambd*penalty[customer][facility_new]
                
                cost_gain = cost_old_pen - cost_new_pen
                
                if cost_gain>max_gain: 
                    max_gain = cost_gain
                    choose_customer = customer
                    choose_facility_old = facility_old
                    choose_facility_new = facility_new
                  
                
        return choose_customer,choose_facility_new,choose_facility_old
    

    def manage_Penalties(self,penalty, cur_sol, facilities_clients):
        
        max_util = 0
        max_customer = 0
        
        for customer, facility in enumerate(cur_sol):
            util = self.matrix_distances[customer][facility]/(1+penalty[customer][facility])
            if max_util < util:
                max_util = util
                max_customer = customer
                
        penalty[customer][cur_sol[customer]] +=1
 
        return penalty
        
    def guided_search_facility(self, ini_sol, time_limit):
        #Control by time! Less than 500
        start = time.time()
        diff = time.time() - start
        
        penalty = [[0 for x in range(len(self.facilities))] for y in range(len(self.customers))] 
        
                ## Parameters
        alpha = 0.1
        lambd = 0
        
        cur_sol = ini_sol.copy()
        cur_sol_dist = self.objective_function(cur_sol)
        #print(cur_sol_dist)
        
        best_sol_dist = cur_sol_dist
        best_sol = cur_sol
        
        facilities_clients = [[ c for c in range(len(cur_sol)) if f.index == cur_sol[c]] for f in self.facilities]
        facilities_available = [f.capacity - sum([ self.customers[c].demand for c in range(len(cur_sol)) if f.index == cur_sol[c]]) for f in self.facilities]
        #print("Here")
        
        #print(facilities_clients)
        #print(facilities_available)
        while diff < time_limit:
            
            customer, facility_new, facility_old = self.select_best_customer(cur_sol, facilities_clients, facilities_available, penalty,lambd)
                
            if customer == -1:
                lambd = alpha * self.objective_function(cur_sol)/ len(ini_sol)
                penalty = self.manage_Penalties(penalty, cur_sol, facilities_clients)
            else:
                
                cost_old = self.matrix_distances[customer][facility_old] + (1 if len(facilities_clients[facility_old]) == 1 else 0) * self.facilities[facility_old].setup_cost
                        
                cost_new = self.matrix_distances[customer][facility_new] + (1 if len(facilities_clients[facility_new]) == 0 else 0) * self.facilities[facility_new].setup_cost
                        
                cost_gain = cost_old - cost_new
                
                cur_sol_dist = cur_sol_dist - cost_gain 
                
                facilities_clients[facility_old].remove(customer)
                facilities_available[facility_old] += self.customers[customer].demand
                facilities_clients[facility_new].append(customer)
                facilities_available[facility_new] -= self.customers[customer].demand
                
                cur_sol[customer] = facility_new
                
            if cur_sol_dist < best_sol_dist:
                best_sol_dist = cur_sol_dist
                best_sol = cur_sol
                
            diff = time.time() - start
            
        print("Done")
        #print(best_sol_dist)
        return self.objective_function(best_sol) , best_sol
    
    def googleSolver(self, time):
        
        result_value_min = 99999999999999999999
        
        n_sub_facilities = 50
        
        distance_matrix_facilities = [[((fa.location.x - fb.location.x) ** 2 + (fa.location.y - fb.location.y) ** 2) ** 0.5 \
                                    for fb in self.facilities]  for fa in self.facilities]
        
        distance_matrix_facilities_indice = [[i for i in range(len(self.facilities))] for j in range(len(self.facilities))]
        
        for i, row in enumerate(distance_matrix_facilities_indice):
            row.sort(key=lambda x: distance_matrix_facilities[i][x])    
        
        
        sub_facilities = np.random.choice(len(self.facilities))
        sub_facilities = distance_matrix_facilities_indice[sub_facilities][:n_sub_facilities]
                
        best_assignment = []
        
        for iterat in range(2):
            print()
            
            solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
            
            sub_assignment = [[solver.IntVar(0.0, 1.0, 'a' + str(i) + ',' + str(j)) for j in range(len(sub_facilities))]  for i in range(len(self.customers))]
            
            sub_facility_open = [solver.IntVar(0.0, 1.0, 'f' + str(j)) for j in range(len(sub_facilities))]
            
            ## Customer assined to only one facility
            for i in range(len(self.customers)):
                solver.Add(sum([sub_assignment[i][j] for j in range(len(sub_facilities))]) == 1)
                
                
            ## Customer assigned to a open facility
            for i in range(len(self.customers)):
                for j in range(len(sub_facilities)):
                    solver.Add(sub_assignment[i][j] <= sub_facility_open[j])
            
            ## Capacity of Warehouse
            
            for j in range(len(sub_facilities)):
                solver.Add(sum([sub_assignment[i][j] * self.customers[i].demand \
                    for i in range(len(self.customers))]) <= self.facilities[sub_facilities[j]].capacity)
    
            objective = solver.Objective()
            
            # Objective: sum all the distance.
            for i in range(len(self.customers)):
                for j in range(len(sub_facilities)):
                    objective.SetCoefficient(sub_assignment[i][j], self.matrix_distances[i][sub_facilities[j]])
                    
            # Objective: sum all the setup cost.
            for j in range(len(sub_facilities)):
                objective.SetCoefficient(sub_facility_open[j], self.facilities[sub_facilities[j]].setup_cost)
    
            objective.SetMinimization()
            
            SEC = 1000
            MIN = 60 * SEC
            solver.SetTimeLimit(1 * MIN)
            
            result_status = solver.Solve()
            print(result_status)
            
            result_value = solver.Objective().Value()
            print(result_value)
            
            if result_status != solver.OPTIMAL and result_status != solver.FEASIBLE:
                continue
            
            if result_status == solver.OPTIMAL:
                break
            
            
            
            assignment_new = []
            
            for i in range(len(self.customers)):
                for j in range(len(sub_facilities)):
                    if sub_assignment[i][j].solution_value() == 1:
                        assignment_new.append(sub_facilities[j])
                        break
        
            if result_value_min> result_value:
                result_value_min = result_value
                best_assignment = assignment_new
        
        return result_value_min, best_assignment


 
 
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    
    #result = opt.run_mip()
    #obj = result[0]
    #solution = result[1]
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    
        
    opt = optimizator(facilities,customers)
    
    obj, ini_sol = opt.initial_solution()
    
    #print(opt.guided_search_facility(ini_sol, 100))
    obj, ini_sol = opt.googleSolver(60)
    #obj, ini_sol = opt.guided_search_facility(ini_sol, 5000)

    solution = ini_sol

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

import os
#os.chdir('G:\\Documentos\\Learning\\Coursera\\DiscreteOptimization\\W6\\facility\\data')
if __name__ == '__main__':
    import sys
    #if True:
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        #file_location = 'fl_16_1'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

