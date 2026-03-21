import random
import copy

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
            if_func(*regparams, **params)
        else:
            else_func(*regparams, **params)
    return NewFunction



def Combine(FuncA,FuncB,*regparams, **params):
    def result(*inregparams, **inparams):
        return FuncA(FuncB(*inregparams, **inparams),*regparams, **params)
    return result


# generates a random population, the least intresting function to modify
def random_population(population_size, individual_size, GoodProb=1/2, *regparams, **params):
    population = list(np.random.choice([0, 1], size=(population_size, individual_size),
                                       replace=True, p=[1 - GoodProb, GoodProb]))
    return population


# simplest possible design of the fitness function
def simplefitness(individual, item_values, item_weights, max_weight, pos_off=1, *regparams, **params):
    return np.dot(individual, item_values)+pos_off if np.dot(individual, item_weights) <= max_weight else pos_off

# gives a low estimate for the ideal fillup value (might be larger then the actual best cost, BUT not the best individuals fitness)
def underflowfitness(individual, item_values, item_weights, max_weight, minratio, *regparams, **params):
    return np.dot(individual, item_values)+minratio*(max_weight-np.dot(individual, item_weights)) if np.dot(individual, item_weights) <= max_weight else 0


# to preserve some of the overflowing choices we can define
# should be lower then if you just remove the overflow
# is affected by order,
def overflowfitness(individual, item_values, item_weights, max_weight, minratio, *regparams, **params):
    ratio = 0
    loss = 0
    overflow = np.dot(individual, item_weights) - max_weight
    i = 0
    while overflow > 0:
        # who needs if conditions (well the max probably has one)
        ratio = max(ratio,individual[i]*item_values[i]/item_weights[i])
        loss += individual[i]*ratio*item_weights[i]
        overflow -= individual[i]*item_weights[i]
        i+=1
        
    return np.dot(individual, item_values)+minratio*(-overflow) - loss


# simplest possible design of the fitness function
def selection(population, fitness_value, number, *regparams, **params):
    return random.choices(population, weights=fitness_value, k=number)

def TournamentSelect(population, fitness_value, number, *regparams, **params):
    pass


# the idea is this is the regular selection algorithm BUT, this scales the fitness values so that the max 
# is 1 and the min is 0 and then lets you apply some function to it.
# note that roulette selection is invariant under scaling of all values
# and that constant shifting makes it more/less uniform (assuming its negative its more uniform and otherwise its less)
def normalalteredselection(population, fitness_value, number, modyf=identity, 
                           normaltop = True, normalbot = True, *regparams, **params):
    bot = (min(fitness_value) if normalbot else 0)
    top = (max(fitness_value) if normaltop else 1)
    if top == 0: # to prevent divide by zero errors
        bot = np.float64(min(bot,-0.001))
        top = -bot
    elif top == bot: # to prevent everything normalizing to zero
        bot = 0
    newfit = [modyf(np.float64(val - bot)/top) for val in fitness_value]
    return random.choices(population, weights=newfit, k=number)


# simplest possible design of the crossing function
def crossover(a,b, *regparams, **params):
    # We randomly choose the crossover index
    crossover_point = random.randint(1, len(a) - 1)

    # np.concatanate creates a new array, hence there is no need for the deepcopying
    newguy = np.concatenate([a[:crossover_point], b[crossover_point:]], axis=0)
    return newguy


def mutation(individual, GoodProb=0.05, *regparams, **params):
    # individual = copy.deepcopy(indiv) 
    # should be no need to copy since we dont care what happens to the old version of the individual
    for j in range(len(individual)):
        if random.random() < GoodProb:
            # This flips 0 to 1 and 1 to 0
            individual[j] = int(not individual[j])
    return individual


class Evolutor:
    def __init__(self, problem_spec=None, /, gen_count=100, Pop_char=(0, 100,0), fitness_func=None,
                 populate_func=None, selector_func=None, cross_func=identity,
                 mutate_func=identity, demutate_func=identity):
        # these are to permit changing the problem later
        self.problem = problem_spec
        self.fitness_b = fitness_func
        self.selector_b = selector_func
        self.crossbreed_b = cross_func
        self.mutate_b = mutate_func
        self.demutate_b = demutate_func
        self.populator_b = populate_func
        
        # this is faster the applying the arguments at runtime
        self.fitness = Apply(fitness_func,problem_spec)
        self.selector = Apply(selector_func,problem_spec)
        self.crossbreed = Apply(cross_func,problem_spec)
        self.mutate = Apply(mutate_func,problem_spec)
        self.demutate = Apply(demutate_func,problem_spec)
        self.populator = Apply(populate_func,problem_spec)

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
            self.gentarget = Apply(self.gentarget,problem_spec)
            self.fitness = Apply(self.fitness,problem_spec)
            self.selector = Apply(self.selector,problem_spec)
            self.crossbreed = Apply(self.crossbreed,problem_spec)
            self.mutate = Apply(self.mutate,problem_spec)
            self.demutate = Apply(self.demutate,problem_spec)
            self.populator = Apply(self.populator,problem_spec)
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
            survivors = copy.deepcopy(self.selector(self.population, self.lastfitness, self.fromprevius))
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


    def Run(self, Silent=False):
        while self.gennumber < self.gentarget:
            self.ProcessGeneration(Silent)
            print("ran gen:", self.gennumber)


    def GiveStats(self):
        return self.totalbest, self.bestsolve, self.fitgraph


def RunExperiment(ProblemSpecs, Specs):
    Experiment = Evolutor(ProblemSpecs,**Specs)
    Experiment.Setup()
    Experiment.Run()
    results = Experiment.GiveStats()
    return simplefitness(results[1], **ProblemSpecs),results[1],results[2]

def loadproblem(filepath, delimiter=","):
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
    minrat = Processed[0][0]
    leng = len(costs)
    sumc = sum(costs)
    sumw = sum(weights)
    maxrat = Processed[-1][0]
    avgrat = sumc/sumw
    problemspecs = {"max_weight" : Limit,
                    "item_weights" : weights,
                    "item_values" : costs, 
                    "individual_size" : leng, 
                    "minratio" : minrat, 
                    "maxratio" : maxrat, 
                    "avgratio" : avgrat,
                    "GoodProb": max((Limit/(sumw/leng))/leng,1/leng)/2
                    }
    return problemspecs


print("Loading_Problems")
problemA = loadproblem(r"03_evolucni_algoritmy\knapsack\debug_10.txt")
problemB = loadproblem(r"03_evolucni_algoritmy\knapsack\debug_20.txt")
problemC = loadproblem(r"03_evolucni_algoritmy\knapsack\input_100.txt")
problemD = loadproblem(r"03_evolucni_algoritmy\knapsack\input_1000.txt")
print("Problems_Lodead")
print(problemD)
testspec = {
    "gen_count" : 500,
    "Pop_char" : (50,200,0),
    "fitness_func" : underflowfitness,
    "populate_func" : random_population, 
    "selector_func" : Apply(normalalteredselection,{"modyf": identity, "normaltop": True, "normalbot" : False}), 
    "cross_func" : crossover,
    "mutate_func" : mutation, 
    "demutate_func" : identity,
}
results = RunExperiment(problemD, testspec)
print(results[0])

# this should hold all varibles along with list of their possible values, and for those list of all their possible specifications
ExperimentVariables = [
    ("Pop_char", [ # just run though all distributions
            ((i,100-i,0),) for i in range(0,100,10)
        ]),
    ("fitness_func", [
            (simplefitness,),
            (underflowfitness,),
            (overflowfitness,),
            (Combine(np.square,simplefitness),),
            (Combine(np.square,underflowfitness),),
            (Combine(np.square,overflowfitness),),
        ]),
    ("populate_func", [
            (random_population,),
        ]),
    ("selector_func", [
            (selection,),
        ]),
    ("cross_func", [
            (crossover,),
        ]),
    ("mutate_func", [
            (identity,),
            (mutation, [ {"bit_mutation_prob": i} for i in 
                        [0.01, 0.02, 0.03, 0.04, 0.05, 0.0625, 0.075, 0.0875, 0.1, 
                         0.125, 0.15, 0.20, 0.225, 0.25, 0.275, 0.3]
            ])
        ]),
    ("demutate_func", [
            (identity,),
        ]),
]

def stringindex(index):
    result = ""
    for i in range(len(index)):
        ind = index[i]
        result += str(ind[0])
        if len(ind) == 2:
            result += "." + str(ind[1])
        if i != len(index)-1:
            result += "-"   
    return result

def makeindex(Varlist):
    index = []
    for i in range(len(Varlist)):
        if len(Varlist[i][1][0])==1:
            index.append((0,))
        elif len(Varlist[i][1][0])==2:
            index.append((0,0))
        else:
            raise("MISTAKE")
    return index

# generates the current index configuration AND pushes the index
def getandpushindex(index, VarList):
    choice = {}
    pushing = True
    for i in range(len(index)):
        Param = VarList[i]
        ind = index[i]
        if len(ind) == 1: # if its an unparametrised element take it directly
            choice[Param[0]] = Param[1][ind[0]][0]
        elif len(ind) == 2:
            choice[Param[0]] = Apply(Param[1][ind[0]][0], Param[1][ind[0]][1][ind[1]])
            if pushing and ind[1] + 1 < len(Param[1][ind[0]][1]):
                index[i] = (ind[0], ind[1] + 1) # push the latter index if possible
                pushing = False
        else:
            raise("MISTAKE")
        
        if pushing:
            if ind[0] + 1 < len(Param[1]): # figure out if we are overflowing
                nexti = ind[0] + 1 
                pushing=False
            else:
                nexti = 0

            # figure out if we need a double or a single index
            if (len(VarList[i][1][nexti]) == 1):
                index[i] = (nexti,)
            elif (len(VarList[i][1][nexti]) == 2):
                index[i] = (nexti,0)
            else:
                raise("MISTAKE")
    return choice, index, pushing

exit()

testind = makeindex(ExperimentVariables)
ok = False
print(testind)
while not ok:
    print(f"runing the {stringindex(testind)} instace")
    choice, testind, ok = getandpushindex(testind,ExperimentVariables)
    pass
print(testind)
print(stringindex(testind))