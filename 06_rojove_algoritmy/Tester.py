from collections import namedtuple
import math
import functools
import csv
import os
import pprint

import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import collections as mc



Vertex = namedtuple("Vertex", ["id", "x", "y", "demand"])

# we know that that we have inifinitly many vehicles of the same size, all routed from the same depot which I normalize as the node of index 0.

class RoutingGraph:
    def __init__(self, file_path):
        self.vertices = []
        self.name = file_path
        self.vehicle_size = 1000
        depot_id = None

        if file_path.lower().endswith(".xml"):
            with open(file_path, encoding="utf-8") as xml_file:
                xml_text = xml_file.read()

            def tag_value(source, tag, start=0):
                opening = f"<{tag}>"
                closing = f"</{tag}>"
                start_tag = source.find(opening, start)
                if start_tag == -1:
                    return None
                value_start = start_tag + len(opening)
                end_tag = source.find(closing, value_start)
                if end_tag == -1:
                    return None
                return source[value_start:end_tag].strip()

            def attr_value(header, attr_name):
                token = f'{attr_name}="'
                attr_start = header.find(token)
                if attr_start == -1:
                    return None
                value_start = attr_start + len(token)
                value_end = header.find('"', value_start)
                if value_end == -1:
                    return None
                return header[value_start:value_end]

            def iter_blocks(source, tag):
                start_token = f"<{tag} "
                end_token = f"</{tag}>"
                search_pos = 0
                while True:
                    block_start = source.find(start_token, search_pos)
                    if block_start == -1:
                        return
                    block_end = source.find(end_token, block_start)
                    if block_end == -1:
                        return
                    open_end = source.find(">", block_start)
                    header = source[block_start:open_end + 1]
                    body = source[open_end + 1:block_end]
                    yield header, body
                    search_pos = block_end + len(end_token)

            instance_name = tag_value(xml_text, "name")
            if instance_name:
                self.name = instance_name

            capacity = tag_value(xml_text, "capacity")
            if capacity is not None:
                self.vehicle_size = float(capacity)

            demands = {}
            for request_header, request_body in iter_blocks(xml_text, "request"):
                node_id = attr_value(request_header, "node")
                quantity = tag_value(request_body, "quantity")
                if node_id is not None and quantity is not None:
                    demands[node_id] = float(quantity)

            nodes = []
            for node_header, node_body in iter_blocks(xml_text, "node"):
                node_id = attr_value(node_header, "id")
                if attr_value(node_header, "type") == "0" and depot_id is None:
                    depot_id = int(node_id)
                elif attr_value(node_header, "type") == "0":
                    raise ValueError(f"Multiple depots found. that bad.")
                    
                x = tag_value(node_body, "cx")
                y = tag_value(node_body, "cy")
                if node_id is not None and x is not None and y is not None:
                    nodes.append((int(node_id), node_id, float(x), float(y)))

            nodes.sort(key=lambda n: n[0])
            for _, node_id, x, y in nodes:
                demand = demands.get(node_id, 0.0)
                if depot_id == int(node_id):
                    depot_id = len(self.vertices) # save position in order
                self.vertices.append(Vertex(node_id, x, y, demand))
                
            if self.vertices:
                depot = self.vertices[depot_id]
                self.vertices[depot_id] = self.vertices[0]
                self.vertices[0] = depot
                self.deposwap = depot_id
                if depot.demand > 0:
                    raise ValueError("Depot cannot have demand greater than 0.")
            return
    
    def __len__(self):
        return len(self.vertices)
    
    def __getitem__(self, key):
        return self.vertices[key]
    
    def __iter__(self):
        return iter(self.vertices)
    
    def __str__(self):
        return f"RoutingGraph with {len(self.vertices)} vertices from {self.name}"
    
    @functools.lru_cache(maxsize=None)
    def distance(self, a, b):
        v1 = self.vertices[a]
        v2 = self.vertices[b]
        return math.sqrt(((v1.x - v2.x) * (v1.x - v2.x)) + ((v1.y - v2.y) * (v1.y - v2.y)))

    def __call__(self, *args):
        if len(args) == 2:
            return self.distance(args[0], args[1])

        if len(args) != 1:
            raise TypeError("RoutingGraph expects Routing(a, b) or Routing(path)")

        path = args[0]
        total_distance = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total_distance += self.distance(a, b)
        return total_distance

    def dist_sum(self):
        return sum(self.distance(0, i) for i in range(1, len(self.vertices)))

aco_rng = np.random.default_rng()

def basecool(a, b, pheromones, Routing, alpha=1, beta=3, *args, **kwargs):
    val = pow(pheromones[a, b], alpha) * pow(1 / Routing.distance(a, b), beta)
    return val if val > 1e-6 else 1e-6
    

# intended for testing purpuses, does nothing.
def identity(primaryinput=None, *args, **kwargs):
    return primaryinput

def examplestag(stagnation, best_gen_length, best_length, stag_threshold=0.05, *args, **kwargs):
    if best_gen_length < best_length * (1 - stag_threshold):
        stagnation *= 0.5 + 0.5 * (best_gen_length / best_length)
    elif best_gen_length > best_length * (1 + stag_threshold):
        stagnation *= 0.5 + 0.5 * (best_gen_length / best_length)
    else:
        stagnation = np.sqrt(stagnation)
    stagnation = max(0.5, stagnation) # prevent LOW stagnation
    stagnation = min(2, stagnation) # prevent HIGH stagnation 
    return stagnation

def roulleteWalker(last, fullfillable, get_cool, *args, **kwargs):
    probs = np.array([get_cool(last, b) for b in fullfillable])
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        probs = np.ones(len(fullfillable), dtype=float) / len(fullfillable)
    else:
        probs = probs / probs_sum
    return aco_rng.choice(fullfillable, p=probs)

def bestWalker(last, fullfillable, get_cool, *args, **kwargs):
    probs = np.array([get_cool(last, b) for b in fullfillable])
    selected = np.argmax(probs)
    return fullfillable[selected]

def fartWalker(last, fullfillable, pheromones, *args, **kwargs):
    selected = max(fullfillable, key=lambda x: pheromones[last, x])
    return selected

def bestrandomWalker(last, fullfillable, get_cool, Top=3, *args, **kwargs):
    probs = np.array([get_cool(last, b) for b in fullfillable])
    best_indices = np.argsort(probs)[-min(Top, len(probs)):] # get top Top indices
    selected = aco_rng.choice(best_indices) # randomly select from Top
    return fullfillable[selected]

def randombestWalker(last, fullfillable, get_cool, number_of_choices=3, *args, **kwargs):
    chosen = aco_rng.choice(fullfillable, size=min(number_of_choices, len(fullfillable)), replace=False)
    probs = np.array([get_cool(last, b) for b in chosen])
    selected = int(np.argmax(probs))
    return chosen[selected]

def devientWalker(last, fullfillable, get_cool, deviance, *args, **kwargs):
    if aco_rng.random() < deviance:
        return roulleteWalker(last, fullfillable, get_cool, *args, **kwargs)
    else:
        return bestWalker(last, fullfillable, get_cool, *args, **kwargs)

def uncertizer(A, B, p, *args, **kwargs):
    if aco_rng.random() < p:
        return A(*args, **kwargs)
    else:
        return B(*args, **kwargs)


def generate_solution(Routing, Walker, return_threshold=0.5, *args, **kwargs):
    number_of_vertices = len(Routing)
    unfullfilled = list(range(1, number_of_vertices))
    solution = [0] # zero index vertex is always the depot
    capacity_left = Routing.vehicle_size
    
    while unfullfilled:
        last = solution[-1]
        fullfillable = [v for v in unfullfilled if Routing[v].demand <= capacity_left]
        closest = min((Routing(last, x) for x in fullfillable), default=np.inf)

        if last != 0 and closest >= Routing(last, 0) * return_threshold:
            fullfillable.append(0) # we can always return to the depot
        elif not fullfillable and unfullfilled:
            raise ValueError("No fullfillable vertices left, but unfullfilled is not empty. This should not happen!")
            
        selected = Walker(last, fullfillable, *args, **kwargs) # Edge selection
        solution.append(selected)
        if selected == 0:
            capacity_left = Routing.vehicle_size
        else:
            capacity_left -= Routing[selected].demand
            unfullfilled.remove(selected)
    if solution[-1] != 0:
        solution.append(0) # return to depot at the end if not already there
        
    return solution

# INTENDED FOR TESTING, VERY NONOPTIMISED
def make_solution(Routing, pheromones, Walker_B=randombestWalker, get_cool_B=basecool, alpha=1, beta=3, return_threshold=0.5, *args, **kwargs):
    data = {
        "pheromones": pheromones,
        "Routing": Routing,
        "alpha": alpha,
        "beta": beta,
        "get_cool_B": get_cool_B,
        "Walker_B": Walker_B,
    }
    data["get_cool"] = functools.partial(get_cool_B, **data)
    data["Walker"] = functools.partial(Walker_B, **data)
    return generate_solution(Routing, data["Walker"], return_threshold=return_threshold, *args, **kwargs)

def generate_ideal_solution(Routing, pheromones, *args, **kwargs):
    walker = functools.partial(fartWalker, pheromones=pheromones)
    return generate_solution(Routing, walker)


# This format is to permit swapping for identity
def TopAntSelect(Ants, Routing, ant_number=10, *args, **kwargs):
    qualities = [Routing(ant) for ant in Ants]
    best_indices = np.argsort(qualities)[:min(ant_number, len(qualities))]
    return [Ants[i] for i in best_indices]

def RandomAntSelect(Ants, ant_number=10, *args, **kwargs):
    return aco_rng.choice(Ants, size=min(len(Ants), ant_number), replace=False)

def RandomHandAntSelect(Ants, Routing, ant_number=10, handsize=5, *args, **kwargs):
    selection = []
    for _ in range(ant_number):
        hand = aco_rng.choice(Ants, size=min(len(Ants), handsize), replace=False)
        qualities = [Routing(ant) for ant in hand]
        best_index = np.argmin(qualities)
        selection.append(hand[best_index])
    return selection

def RoulleteAntSelect(Ants, Routing, ant_number=10, Postprocessing = np.square, *args, **kwargs):
    qualities = [Postprocessing(Routing(ant)) for ant in Ants]
    best_indices = aco_rng.choice(len(Ants), size=min(ant_number, len(Ants)), replace=False, p=qualities/np.sum(qualities))
    return [Ants[i] for i in best_indices]

def OptimiseGroup(Group, Routing, *args, **kwargs):
    route = list(Group)
    if len(route) <= 4:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                before = Routing(route[i - 1], route[i]) + Routing(route[j], route[j + 1])
                after = Routing(route[i - 1], route[j]) + Routing(route[i], route[j + 1])

                if after < before:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return route

def OptimisePath(Solution, Routing, *args, **kwargs):
    groups = [[0]]
    for node in Solution[1:-1]:
        groups[-1].append(node)
        if node == 0:
            groups.append([0])
    groups[-1].append(Solution[-1])

    # Keep OptimiseGroup as an optional override if provided later.
    optgroups = []
    for group in groups:
        optgroups.append(OptimiseGroup(group, Routing))

    optsolve = [0]
    for group in optgroups:
        if group[0] != 0:
            group = [0] + group
        if group[-1] != 0:
            group = group + [0]
        optsolve.extend(group[1:])

    if optsolve[-1] != 0:
        optsolve.append(0)

    return optsolve

class Ant_Solver:
    def __init__(self):
        pass

    def setup(self, routs, number_of_ants=50, max_iterations=50, alpha=1, beta=3, Q=100, t_decay=0.8, p_decay=0.25, init_pheromone=0.01, return_threshold=0.5, 
              ant_walker=roulleteWalker, ant_helper = identity, ant_select = identity, optimise_point = 1, *args, **kwargs):
        self.Routing = routs
        self.number_of_ants = number_of_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.t_decay = t_decay
        self.p_decay = p_decay
        self.initpher = init_pheromone
        self.best_solution = None
        self.best_length = routs.dist_sum() * 2 # approximates a trivial solution
        self.fitgraph = list()
        self.return_threshold = return_threshold
        self.ant_walker_base = ant_walker
        self.ant_helper_base = ant_helper
        self.ant_select_base = ant_select
        self.optimise_point = optimise_point
        self.data = {
            "Routing": routs,
            "number_of_ants": number_of_ants,
            "max_iterations": max_iterations,
            "alpha": alpha,
            "beta": beta,
            "Q": Q,
            "t_decay": t_decay,
            "p_decay": p_decay,
            "init_pheromone": init_pheromone,
            "return_threshold": return_threshold,
            "optimise_point": optimise_point,
        }
        copycat = functools.partial(basecool, **self.data)
        self.get_cool = lambda a, b: copycat(a,b, self.pheromones)
        self.data["get_cool"] = self.get_cool
        self.Walker = functools.partial(ant_walker, **self.data)
        self.data["Walker"] = self.Walker
        self.generate_solution = functools.partial(generate_solution, **self.data)
        self.data["generate_solution"] = self.generate_solution
        self.Helper = functools.partial(ant_helper, **self.data)
        self.data["Helper"] = self.Helper
        self.Selecter = functools.partial(ant_select, **self.data)
        self.data["Selecter"] = self.Selecter

        self.setup_pheromone()

    def setup_pheromone(self):
        N = len(self.Routing)
        self.pheromones = self.initpher * np.ones(shape=(N,N))
    
    def update_pheromone(self, solutions):
        pheromone_update = np.zeros(shape=self.pheromones.shape)
        for solution in solutions:
            solution_length = self.Routing(solution)
            for a, b in zip(solution[:-1], solution[1:]):
                pheromone_update[a, b] += self.Q / solution_length
        
        self.pheromones = (1 - self.t_decay) * self.pheromones + pheromone_update/(len(solutions))*10
        # the scaling is to make Q and the amount of ants picked a more independent parameter
    
    def generate_generation(self):
        for _ in range(self.number_of_ants):
            solv = self.generate_solution()
            for a, b in zip(solv[:-1], solv[1:]):
                self.pheromones[a, b] *= (1 - self.p_decay) # pheromone decay on the path of the generated solution
            yield solv
    
    def validate_solution(self, solution):
        return solution[0] == 0 and \
        all(self.Routing[solution[i]].demand <= self.Routing.vehicle_size for i in range(1, len(solution))) and \
        all(solution[i] != solution[i + 1] for i in range(len(solution) - 1))

    def run_generation(self):
        unchecked_solutions = list(self.generate_generation())
        solutions = [solution for solution in unchecked_solutions if self.validate_solution(solution)]
        if not solutions:
            raise ValueError("No valid solutions generated in this generation.")

        if self.optimise_point == 0:
            solutions = [self.Helper(solution) for solution in solutions]
        solution_lengths = [self.Routing(solution) for solution in solutions]
        good_solves = self.Selecter(solutions, Routing=self.Routing)
        if self.optimise_point == 1:
            good_solves = [self.Helper(solution) for solution in good_solves]
        self.update_pheromone(good_solves)
        best_gen_solution = None
        best_gen_length = float("inf")

        for candidate_solution, candidate_length in zip(solutions, solution_lengths):
            if candidate_length < best_gen_length:
                best_gen_length = candidate_length
                best_gen_solution = candidate_solution


        if best_gen_length < self.best_length:
            self.best_length = best_gen_length
            self.best_solution = best_gen_solution

        self.fitgraph.append(best_gen_length)
        
        return np.min(solution_lengths), np.mean(solution_lengths), np.max(solution_lengths)
    
    def run(self, log_rate = 0):
        runmin = float("inf")
        runsum = float(0)
        runmax = float("-inf")
        if log_rate != 0:
            print("Iteration\tMinimum value\tMean value\tMaximum value")
        for i in range(self.max_iterations):
            min_val, mean_val, max_val = self.run_generation()
            runmin = min(runmin, min_val)
            runsum += mean_val
            runmax = max(runmax, max_val)
            if log_rate != 0 and ((i+1) % log_rate == 0):
                print(f"{i+1:8}:\t{runmin:5.8f}\t{runsum/log_rate:5.8f}\t{runmax:5.8f}")
                runmin = float("inf")
                runsum = float(0)
                runmax = float("-inf")
            elif log_rate != 0 and i+1 == self.max_iterations:
                print(f"{i+1:8}:\t{runmin:5.8f}\t{runsum/(i+1%log_rate):5.8f}\t{runmax:5.8f}")

    def giveStats(self):
        return self.best_solution, self.pheromones, self.fitgraph

def RunExperiment(ProblemSpecs, Specs, report_interval=10):
    Experiment = Ant_Solver()
    Experiment.setup(ProblemSpecs, **Specs)
    Experiment.run(log_rate=report_interval)
    best_solution, pheromones, fitgraph = Experiment.giveStats()
    best_distance = ProblemSpecs(best_solution)
    return best_distance, best_solution, fitgraph, pheromones

def stringindex(index):
    result = ""
    for i in range(len(index)):
        result += str(index[i])
        if i != len(index)-1:
            result += "."   
    return result

def makeindex(Varlist):
    index = [0 for _ in range(len(Varlist))]
    return index

def indexsize(Varlist):
    index = 1
    for i in range(len(Varlist)):
        index *= len(Varlist[i][1])
    return index

# generates the current index configuration AND pushes the index
def getandpushindex(index, VarList, Push=True):
    choice = {}
    pushing = Push
    for i in range(len(index)):
        Param = VarList[i]
        ind = index[i]

        choice[Param[0]] = Param[1][ind]
        if pushing:
            if ind + 1 < len(Param[1]): # figure out if we are overflowing
                index[i] = ind + 1 
                pushing=False
            else:
                index[i] = 0
    return choice, index, pushing

def AnaliseConfiguration(ProblemSpecs, Specs, num = 10, nameprefix = ""):
    dataset = []
    printed = 0
    print("_"*20)
    for iter in range(num):
        start_time = time.perf_counter()  # High-precision timer
        score, _, data, Pher = RunExperiment(ProblemSpecs, Specs, 0)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        dataset.append(
            {"index": nameprefix + str(iter), 
             "score": score, 
             "data": data, 
             "final": data[-1],
             "time": run_time,
             "pheromones": Pher})
        if ((iter+1)*20)//num > printed:
            print("#"*((((iter+1)*20)//num)-printed), end="", flush=True) #super basic progress bar
            printed = ((iter+1)*20)//num
    print()
    return dataset

def MassExperiment(ProblemSpecs, VarList):
    dataset = []
    testind = makeindex(VarList)
    ok = False
    excount = indexsize(VarList)
    i = 0
    printed = 0
    print("_"*20)
    while not ok:
        #print(f"runing the {stringindex(testind)} instance")
        choice, testind, ok = getandpushindex(testind,VarList)

        start_time = time.perf_counter()  # High-precision timer
        score, _, data, Pher = RunExperiment(ProblemSpecs, choice, 0)
        end_time = time.perf_counter()

        run_time = end_time - start_time
        dataset.append(
            {"index": stringindex(testind), 
             "score": score, 
             "data": data, 
             "final": data[-1],
             "time": run_time,
             "pheromones": Pher}
        )
        i += 1
        if (i*20)//excount > printed:
            print("#"*(((i*20)//excount)-printed),end="", flush=True) #super basic progress bar
            printed = (i*20)//excount
    print()
    return dataset

def ProcessData(dataset):
    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset)
    df_display = df.drop('data', axis=1) if 'data' in df.columns else df

    df_sorted = df_display.sort_values('score', ascending=False)
    print("\nExperiment Results Sorted by Score:")
    print(df_sorted.to_string(index=False))

    print(f"\nSummary Statistics:")
    print(f"Total Experiments: {len(df)}")
    print(f"Best Score: {df['score'].max()}")
    print(f"Worst Score: {df['score'].min()}")
    print(f"Average Score: {df['score'].mean():.2f}")
    print(f"Average Time: {df['time'].mean():.2f}s")
    return df

def ShowData(df):
    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot each experiment's fitness evolution
    for i, row in df.iterrows():
        plt.plot(row['data'], label=f"Config {row['index']} (Score: {row['score']})", alpha=0.7)

    # Customize the plot
    plt.title('Fitness Evolution Across All Experiments', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Optional: Create a summary plot showing only best/worst/average
    plt.figure(figsize=(10, 6))

    # Calculate average fitness per generation
    all_data = [row['data'] for _, row in df.iterrows()]
    min_length = min(len(d) for d in all_data)
    truncated_data = [d[:min_length] for d in all_data]

    avg_fitness = np.mean(truncated_data, axis=0)
    best_fitness = np.max(truncated_data, axis=0)
    worst_fitness = np.min(truncated_data, axis=0)

    plt.plot(avg_fitness, label='Average Fitness', linewidth=2, color='blue')
    plt.plot(best_fitness, label='Best Fitness', linewidth=2, color='green')
    plt.plot(worst_fitness, label='Worst Fitness', linewidth=2, color='red')

    plt.fill_between(range(len(avg_fitness)), worst_fitness, best_fitness, alpha=0.2, color='blue')

    plt.title('Fitness Evolution Summary', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.scatter(df['time'], df['score'], alpha=0.6)

    # Add labels for each point
    for i, row in df.iterrows():
        plt.annotate(row['index'], (row['time'], row['score']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title('Score vs Execution Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Final Score')
    plt.grid(True, alpha=0.3)
    plt.show()

def show_solution(routing_graph, solution):
    vertices = routing_graph.vertices
    x = [vertices[i].x for i in solution]
    y = [vertices[i].y for i in solution]
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, marker='o')
    plt.title(f"Solution with distance {routing_graph(solution):.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.show()
    
def plot_pheromones(routing_graph, pheromones, solution=None):
    vertices = routing_graph.vertices
    if solution is None:
        solution = generate_ideal_solution(routing_graph, pheromones, 0)
    # Render pheromones (blue, line width corresponds to the pheromon value on the edge)
    lines = []
    colors = []
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            lines.append([(v1.x, v1.y), (v2.x, v2.y)])
            colors.append(pheromones[i, j])

    lc = mc.LineCollection(lines, linewidths=np.array(colors))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    
    # Print ideal solution's length
    plt.title(f"Solution length: {routing_graph(solution):.2f}")

    # Render the solution (red)
    solution_lines = []
    for i, j in zip(solution, solution[1:] + solution[0:1]):
        solution_lines.append([(vertices[i].x, vertices[i].y), (vertices[j].x, vertices[j].y)])

    solutions_lc = mc.LineCollection(solution_lines, colors="red")

    ax.add_collection(solutions_lc)

    plt.show()
    
def plot_graph(fit_graph):
    plt.plot(fit_graph)
    plt.xlabel("Generation")
    plt.ylabel("Best solution length")
    plt.title("Best solution length over iterations")
    plt.grid()
    plt.show()


Good_Specs_C = {
    "number_of_ants":   25,
    "max_iterations":   25,
    "alpha":            2,
    "beta":             3,
    "Q":                150,
    "t_decay":          0.8,
    "p_decay":          0.20,
    "init_pheromone":   0.01,
    "return_threshold": 0.1,
    "ant_walker":       functools.partial(devientWalker, deviance=0.1), # roulleteWalker,
    "ant_helper":       OptimisePath,
    "ant_select":       functools.partial(TopAntSelect, ant_select=10),
    "optimise_point":   1
}
Good_Specs_AB = {
    "number_of_ants":   50,
    "max_iterations":   50,
    "alpha":            2,
    "beta":             3,
    "Q":                150,
    "t_decay":          0.8,
    "p_decay":          0.20,
    "init_pheromone":   0.01,
    "return_threshold": 0,
    "ant_walker":       functools.partial(devientWalker, deviance=0.1), # roulleteWalker,
    "ant_helper":       OptimisePath,
    "ant_select":       functools.partial(TopAntSelect, ant_select=10),
    "optimise_point":   1
}


ExperimentVariables = [
    ("number_of_ants", [
            10,
        ]),
    ("max_iterations", [
            100
        ]),
    ("alpha",
            list(range(1, 6, 1))
        ),
    ("beta",
            list(range(1, 6, 1))
        ),
    ("Q",
            list(range(50, 300, 50))
        ),
    ("t_decay", 
            [ 1 - 0.05*i for i in range(5)]
        ),
    ("p_decay", 
            [ 0.75 + 0.05*i for i in range(5)]
        ),
    ("init_pheromone", [ 
            0.01 
        ]),
    ("return_threshold", 
            list(i/10 for i in range(11))
        ),
    ("ant_walker", [ 
            functools.partial(devientWalker, deviance=0.1),
        ]),
    ("ant_helper", [ 
            OptimisePath,
        ]),
    ("ant_select", [ 
            functools.partial(TopAntSelect, ant_select=10),
        ]),
    ("optimise_point", [ 
            1 
        ]),
]

Experimentspecs = {
    "number_of_ants":   30,
    "max_iterations":   20,
    "alpha":            2,
    "beta":             3,
    "Q":                150,
    "t_decay":          0.8,
    "p_decay":          0.20,
    "init_pheromone":   0.01,
    "return_threshold": 0.1,
    "ant_walker":       functools.partial(devientWalker, deviance=0.1), # roulleteWalker,
    "ant_helper":       OptimisePath,
    "ant_select":       functools.partial(TopAntSelect, ant_select=10),
    "optimise_point":   1
}

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    print("Loading_Problems")
    RouteA = RoutingGraph(os.path.join(base_dir, "routing", "data_32.xml"))
    print(str(RouteA))
    RouteB = RoutingGraph(os.path.join(base_dir, "routing", "data_72.xml"))
    RouteC = RoutingGraph(os.path.join(base_dir, "routing", "data_422.xml"))
    print("Problems_Lodead")

    Routs = RouteA

    best_length, best_solution, log_of_best_distances, pheromones = \
        RunExperiment(RouteA, Good_Specs_AB, report_interval=5)
    print(f"Best solution found: {best_solution} with length {best_length:.2f}")
    
    plot_graph(log_of_best_distances)
    show_solution(Routs, best_solution)
    
    plot_pheromones(Routs, pheromones)
    plot_pheromones(Routs, pheromones, best_solution)



    dataset = AnaliseConfiguration(RouteA, Experimentspecs)
    df = ProcessData(dataset)
    ShowData(df)