import random
import functools
import os
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# intended for testing purpuses, does nothing.
def identity(primaryinput=None, *regparams, **params):
    return primaryinput


# applies some arguments to a generalized function
def Apply(func, Args):
    return lambda *x, **y: func(*x, **Args, **y)


# takes a pair of functions and makes the input either be passed to the first with a prob likelyhood or into the latter
def Probate(if_func, prob=1/2, else_func=identity):
    def NewFunction(*regparams, **params):
        if random.random() < prob:
            return if_func(*regparams, **params)
        else:
            return else_func(*regparams, **params)
    return NewFunction



def Combine(FuncA, FuncB, *regparams, **params):
    partial_A = functools.partial(FuncA, *regparams, **params)
    return lambda *inregparams, **inparams: partial_A(FuncB(*inregparams, **inparams))


# generates a random population, the least intresting function to modify
def random_population(population_size, individual_size, GoodProb=1/2, *regparams, **params):
    population = np.random.choice([0, 1], size=(population_size, individual_size),
                                   replace=True, p=[1 - GoodProb, GoodProb])
    return [row for row in population]

# should generate the population by building up untill it would overflow the limit.
def smart_random_population(population_size, individual_size, max_weight, item_weights, *regparams, **params):
    population = []
    for _ in range(population_size):
        individual = np.zeros(individual_size, dtype=int)
        current_weight = 0
        item_idx = -1
        while current_weight <= max_weight:
            item_idx = random.randint(0, individual_size - 1)
            current_weight += item_weights[item_idx] * (1-individual[item_idx])
            individual[item_idx] = 1
        individual[item_idx] = 0 # remove the item that caused the overflow
        population.append(individual)
    return population


# A slightly different version I was intrested in testing.
def smart_random_population_b(population_size, individual_size, max_weight, item_weights, *regparams, **params):
    population = []
    for _ in range(population_size):
        individual = np.zeros(individual_size, dtype=int)
        current_weight = 0
        item_idx = -1
        while current_weight <= max_weight:
            item_idx = random.randint(0, individual_size - 1)
            current_weight += item_weights[item_idx] # * (1-individual[item_idx])
            individual[item_idx] = 1
        individual[item_idx] = 0 # remove the item that caused the overflow
        population.append(individual)
    return population


# simplest possible design of the fitness function
def simplefitness(individual, item_values, item_weights, max_weight, pos_off=0, *regparams, **params):
    weight = np.dot(individual, item_weights)
    value = np.dot(individual, item_values)
    return value + pos_off if weight <= max_weight else pos_off


# gives a low estimate for the ideal fillup value (might be larger then the actual best cost, BUT not the best individuals fitness)
def underflowfitness(individual, item_values, item_weights, max_weight, minratio, *regparams, **params):
    weightsum = np.dot(individual, item_weights)
    value = np.dot(individual, item_values)
    return value+minratio*(max_weight-weightsum) if weightsum <= max_weight else 0


# to preserve some of the overflowing choices we can define
# should be lower then if you just remove the overflow
# is affected by order,
def overflowfitnessOLD(individual, item_values, item_weights, max_weight, minratio, maxratio, *regparams, **params):
    value = np.dot(individual, item_values)
    overflow = np.dot(individual, item_weights) - max_weight
    i = 0
    while overflow > 0:
        value -= individual[i]*item_values[i] * 1.5 # the multiplier is to actually make it good to remove them extra items
        overflow -= individual[i]*item_weights[i]
        i+=1
    return max(0,value+minratio*(-overflow))

# to preserve some of the overflowing choices we can define
# should be lower then if you just remove the overflow
# is affected by order,
# had very bad results, not sure why but kept for records
def overflowfitness(individual, item_values, item_weights, max_weight, minratio, maxratio, *regparams, **params):
    value = np.dot(individual, item_values)
    weight = np.dot(individual, item_weights)
    overflow = weight - max_weight
    if overflow < 0:
        value += minratio * -overflow
    else: # overflow > 0:
        value -= maxratio * overflow
    return value


# simplest possible design of the fitness function
def selection(population, fitness_value, number, *regparams, **params):
    return random.choices(population, weights=fitness_value, k=number)

def base_indiv_comparator(tournament, fitness_value, population, *regparams, **params):
    return max(tournament, key=lambda i: fitness_value[i])

# this still uses the computed fitness values, but it can be somewhat easily modified to use a different system
def TournamentSelect(population, fitness_value, number, tournament_size=3, tournament_func=base_indiv_comparator, *regparams, **params):
    selected = []
    for _ in range(number):
        tournament = random.sample(range(len(population)), tournament_size)
        winner = tournament_func(tournament,fitness_value, population)
        selected.append(population[winner])
    return selected


# the idea is this is the regular selection algorithm BUT, this scales the fitness values so that the max 
# is 1 and the min is 0 and then lets you apply some function to it.
# note that roulette selection is invariant under scaling of all values
# and that constant shifting makes it more/less uniform (assuming its negative its more uniform and otherwise its less)
def normalalteredselection(population, fitness_value, number, modyf=identity, 
                           normaltop = True, normalbot = True, *regparams, **params):
    bot = (min(fitness_value) if normalbot else 0)
    top = (max(fitness_value)-bot if normaltop else 1)
    if top == 0: # to prevent divide by zero errors
        top = 1
        bot -= 1
    fv = np.array(fitness_value, dtype=np.float64)
    newfit = modyf((fv - bot) / top)
    #newfit = [modyf((val - bot) / top) for val in fitness_value]
    return random.choices(population, weights=newfit, k=number)


# simplest possible design of the crossing function
def crossover(a,b, *regparams, **params):
    # We randomly choose the crossover index
    crossover_point = random.randint(1, len(a) - 1)

    # np.concatanate creates a new array, hence there is no need for the deepcopying
    newguy = np.concatenate([a[:crossover_point], b[crossover_point:]], axis=0)
    return newguy


def uniform_crossover(a, b, swap_prob=0.5, *regparams, **params):
    a_arr = np.array(a, copy=False)
    b_arr = np.array(b, copy=False)
    mask = np.random.rand(a_arr.shape[0]) < swap_prob
    child = np.where(mask, a_arr, b_arr)
    return child


def mutation(individual, GoodProb=0.05, *regparams, **params):
    # Create a copy to avoid in-place modification of parent individuals
    ind = np.array(individual, copy=True)
    for j in range(len(ind)):
        if random.random() < GoodProb:
            # This flips 0 to 1 and 1 to 0
            ind[j] = int(not ind[j])
    return ind


def mutation(individual, GoodProb=0.05, *regparams, **params):
    # Create a copy to avoid in-place modification of parent individuals
    ind = np.array(individual, copy=True)
    for j in range(len(ind)):
        if random.random() < GoodProb:
            # This flips 0 to 1 and 1 to 0
            ind[j] = int(not ind[j])
    return ind


def mutation_b(individual, individual_size, continue_prob=0.5, *regparams, **params):
    # Create a copy to avoid in-place modification of parent individuals
    ind = np.array(individual, copy=True)
    flip_ind = random.randint(0, individual_size - 1)
    while random.random() < continue_prob:
        ind[flip_ind] = int(not ind[flip_ind])
    return ind


def mutation_c(individual, individual_size, GoodProb=0.05, allowence = 1, *regparams, **params):
    # Create a copy to avoid in-place modification of parent individuals
    ind = np.array(individual, copy=True)
    patiance = np.ceil(np.sqrt(individual_size)) + 5
    if random.random() < GoodProb:
        allowence += 1
    while allowence > 0 and patiance > 0:
        patiance -= 1
        flip_ind = random.randint(0, individual_size - 1)   
        # we either take a bit or we leave it there
        allowence +=  2 * individual[flip_ind] - 1
        ind[flip_ind] = int(not ind[flip_ind])
    return ind


def demutate(individual, individual_size, max_weight, item_weights, Continue_Prob=0.75, *regparams, **params):
    ind = np.array(individual, copy=True)
    weight = np.dot(ind, item_weights)
    while random.random() < Continue_Prob and weight > max_weight:
        i = random.randint(0, individual_size - 1)
        weight -= item_weights[i]*ind[i] # remove the item from weight count
        ind[i] = 0 # remove the item from the individual
    return ind


def demutate_b(individual, individual_size, max_weight, item_weights, GoodProb=0.5, *regparams, **params):
    ind = np.array(individual, copy=True)
    weight = np.dot(ind, item_weights)
    Continue_Prob = 1-np.sqrt(GoodProb)
    while random.random() < Continue_Prob and weight > max_weight:
        i = random.randint(0, individual_size - 1)
        weight -= item_weights[i]*ind[i] # remove the item from weight count
        ind[i] = 0 # remove the item from the individual
    return ind


def demutate_c(individual, individual_size, max_weight, item_weights, GoodProb=0.75, *regparams, **params):
    ind = np.array(individual, copy=True)
    weight = np.dot(ind, item_weights)
    Continue_Prob = 1-np.sqrt(GoodProb)
    selected_indices = np.where(ind == 1)[0]
    while random.random() < Continue_Prob and weight > max_weight:
        i = np.random.choice(selected_indices)
        weight -= item_weights[i]*ind[i] # remove the item from weight count
        ind[i] = 0 # remove the item from the individual
        # yes the ability to select items twice is recognized
        # I am lazy and it doesnt break anything
    return ind


class Evolutor:
    def __init__(self, problem_spec, *, gen_count=100, Pop_char=(0, 100,0), fitness_func=simplefitness,
                 populate_func=random_population, selector_func=selection, cross_func=crossover,
                 mutate_func=mutation, demutate_func=identity):
        # save the problem specs
        self.problem = problem_spec

        # these are to permit changing the problem later
        self.fitness_b = fitness_func
        self.selector_b = selector_func
        self.crossbreed_b = cross_func
        self.mutate_b = mutate_func
        self.demutate_b = demutate_func
        self.populator_b = populate_func
        
        # Use functools.partial for cleaner, faster partial application
        self.fitness    = functools.partial(fitness_func, **problem_spec)
        self.selector   = functools.partial(selector_func, **problem_spec)
        self.crossbreed = functools.partial(cross_func, **problem_spec)
        self.mutate     = functools.partial(mutate_func, **problem_spec)
        self.demutate   = functools.partial(demutate_func, **problem_spec)
        self.populator  = functools.partial(populate_func, **problem_spec)

        self.gentarget = gen_count
        self.populationchar = Pop_char
        self.population = []
        self.lastfitness = []
        self.totalbest = 0
        self.bestsolve = None
        self.lastbest = 0
        self.gennumber = 0
        self.fitgraph = []
        self.popsize = sum(Pop_char)-Pop_char[2]    # fit to params
        self.fromprevius = Pop_char[0]  # fit to params
        self.nextgen = Pop_char[1]      # fit to params
        self.elite = 0 #Pop_char[2]        # fit to params # Disabled untill I feel like implenting it


    def Setup(self, problem_spec =None):
        if problem_spec != None:
            self.problem = problem_spec
            # dont forget to rewrite the functions to the new specification
            self.fitness    = functools.partial(self.fitness,**problem_spec)
            self.selector   = functools.partial(self.selector,**problem_spec)
            self.crossbreed = functools.partial(self.crossbreed,**problem_spec)
            self.mutate     = functools.partial(self.mutate,**problem_spec)
            self.demutate   = functools.partial(self.demutate,**problem_spec)
            self.populator  = functools.partial(self.populator,**problem_spec)
        self.population = self.populator(self.popsize)
        self.lastfitness = list(map(self.fitness, self.population))
        bestguy = np.argmax(self.lastfitness)
        self.lastbest = self.lastfitness[bestguy]
        self.bestsolve = self.population[bestguy]
        self.totalbest = self.lastbest
        self.fitgraph = [self.lastbest]


    def ProcessGeneration(self, Silent=False):
        # Find who is what
        try:
            parents = self.selector(self.population, self.lastfitness, self.nextgen)
            survivors = self.selector(self.population, self.lastfitness, self.fromprevius)
        except ValueError:
            print(self.population)
            print(np.sum(self.lastfitness))
            raise(ValueError)
        children = [self.crossbreed(parents[i], parents[i-1]) for i in range(len(parents))]
        PreMutation = survivors + children #+ elites 
        PostMutation = list(map(self.mutate, PreMutation))
        #elites = [self.population[i] for i in np.argpartition(self.lastfitness, -self.elite)[-self.elite:]]

        self.population = PostMutation #+ Elites
        self.lastfitness = list(map(self.fitness, self.population))

        # Logging subroutine
        self.gennumber += 1
        bestguy = np.argmax(self.lastfitness)
        self.lastbest = self.lastfitness[bestguy]
        if (self.lastbest > self.totalbest):
            self.bestsolve = self.population[bestguy]
            self.totalbest = self.lastbest
        if Silent:
            self.fitgraph[-1] = self.lastbest # we still want the graph to reflect states between logging.
        else:
            self.fitgraph.append(self.lastbest)


    def Run(self, Silent=False, ReportInterval=10):
        while self.gennumber < self.gentarget:
            self.ProcessGeneration(Silent)
            if not Silent and ReportInterval >0 and self.gennumber % ReportInterval == 0:
                print(f"Generation {self.gennumber} best fitness: {self.lastbest} total best: {self.totalbest}")


    def GiveStats(self):
        return self.totalbest, self.bestsolve, self.fitgraph


def RunExperiment(ProblemSpecs, Specs, report_interval=10):
    Experiment = Evolutor(ProblemSpecs,**Specs)
    Experiment.Setup()
    Experiment.Run(ReportInterval=report_interval)
    results = Experiment.GiveStats()
    return simplefitness(results[1], **ProblemSpecs),results[1],results[2]

def loadproblem(filepath):
    data = []
    Limit = 0
    print(repr(filepath))
    with open(filepath, "r") as file:
        Limit = int(file.readline().split()[1])
        for line in file:
            line = line.strip()

            parts = line.split()
            cost = int(parts[0])
            weight = int(parts[1])

            data.append((cost / weight,cost, weight))
    
    Processed = sorted(data)
    weights = [triple[2] for triple in Processed]
    costs = [triple[1] for triple in Processed]
    ratios = [triple[0] for triple in Processed]
    leng = len(costs)
    sumc = sum(costs)
    sumw = sum(weights)
    minrat = Processed[0][0]
    maxrat = Processed[-1][0]
    avgrat = sumc/sumw
    medrat = np.median(ratios)
    problemspecs = {"max_weight" : Limit,
                    "item_weights" : weights,
                    "item_values" : costs, 
                    "individual_size" : leng, 
                    "minratio" : minrat, 
                    "maxratio" : maxrat, 
                    "avgratio" : avgrat,
                    "medratio" : medrat,
                    "GoodProb": (np.float64(1)/leng)/4 # max(np.float64(np.median(weights))/Limit,1/leng)/2
                    }
    return problemspecs
# Problem Definition Dictionary Includes:
#   - max_weight        : the total weight limit
#   - item_weights      : the list of all item weights (sorted by their ratio)
#   - item_values       : the list of all item costs (sorted by their ratio)
#   - individual_size   : the total amount of items (ergo individual count)
#   - minratio          : the average of the ratio of cost to weight
#   - maxratio          : the average of the ratio of cost to weight
#   - avgratio          : the average of the ratio of cost to weight
#   - medratio          : the median  of the ratio of cost to weight
#   - GoodProb          : very a rough estimate for how likely a bit is in the solution
# do note that the good prob is literally just 1/individual_size scaled by a constant


ExperimentVariables = [
    ("gen_count", [
            100,
        ]),
    ("Pop_char", [ # just run though all intresting distributions
            (i,100-i,0) for i in range(0,50,25)
        ]),
    ("fitness_func", [
            simplefitness,
            underflowfitness,
            #overflowfitness,
        ]),
    ("populate_func", [
            #random_population,
            smart_random_population_b,
        ]),
    ("selector_func", [
            #selection,
            normalalteredselection,
            functools.partial(normalalteredselection, modyf=np.square),
            TournamentSelect,
        ]),
    ("cross_func", [
            crossover,
            uniform_crossover,
        ]),
    ("mutate_func", [
            identity,
            mutation,
            mutation_b,
            mutation_c,
        ]),
    ("demutate_func", [
            identity,
            demutate,
            demutate_b,
        ]),
]

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

def AnaliseConfiguration(ProblemSpecs, Specs, num = 10, nameprefix = ""):
    dataset = []
    printed = 0
    print("_"*20)
    for iter in range(num):
        start_time = time.perf_counter()  # High-precision timer
        score, bestguy, data = RunExperiment(ProblemSpecs, Specs, 0)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        dataset.append(
            {"index": nameprefix + str(iter), 
             "score": score, 
             "data": data, 
             "final": data[-1],
             "time": run_time})
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
        score, bestguy, data = RunExperiment(ProblemSpecs, choice, 0)
        end_time = time.perf_counter()

        run_time = end_time - start_time
        dataset.append(
            {"index": stringindex(testind), 
             "score": score, 
             "data": data, 
             "final": data[-1],
            "time": run_time})
        i += 1
        if (i*20)//excount > printed:
            print("#"*(((i*20)//excount)-printed),end="", flush=True) #super basic progress bar
            printed = (i*20)//excount
    print()
    return dataset

GeneralConfiguration = [
    ("gen_count", [
            100,
        ]),
    ("Pop_char", [ # just run though all intresting distributions
            (i,100-i,0) for i in range(0,50,25)
        ]),
    ("fitness_func", [
            simplefitness,
            underflowfitness,
            #overflowfitness,       # carefull this has negative fit values
        ]),
    ("populate_func", [
            random_population,
            smart_random_population,
            smart_random_population_b,
        ]),
    ("selector_func", [
            #selection,              # carefull this breaks when non-positive fit values
            normalalteredselection,
            functools.partial(normalalteredselection, modyf=np.square),
            TournamentSelect,
        ]),
    ("cross_func", [
            crossover,
            uniform_crossover,
        ]),
    ("mutate_func", [
            #identity,
            mutation,
            mutation_b,
            mutation_c,
        ]),
    ("demutate_func", [
            identity,
            demutate,
            demutate_b,
        ]),
]

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    print("Loading_Problems")
    problemA = loadproblem(os.path.join(base_dir, "knapsack", "debug_10.txt"))
    problemB = loadproblem(os.path.join(base_dir, "knapsack", "debug_20.txt"))
    problemC = loadproblem(os.path.join(base_dir, "knapsack", "input_100.txt"))
    problemD = loadproblem(os.path.join(base_dir, "knapsack", "input_1000.txt"))
    print("Problems_Lodead")

    
    testspec = {
        "gen_count" : 150,
        "Pop_char" : (25,125,0),
        "fitness_func" : simplefitness,
        "populate_func" : smart_random_population_b, 
        "selector_func" : TournamentSelect, 
        "cross_func" : uniform_crossover,
        "mutate_func" : mutation_b, 
        "demutate_func" : demutate_b,
    }



    dataset = AnaliseConfiguration(problemD, testspec, num=10)
    df = ProcessData(dataset)
    ShowData(df)




exit()
