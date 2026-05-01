import time
import itertools
import random
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def rzip(left,right):
    return zip(left, itertools.chain(right,itertools.repeat(None)))

def TryGet(d, k, fail=None, skip=None):
    if k == skip:
        return fail
    try:
        return d[k]
    except KeyError:
        return fail

def GetSeed(): # returns the value associated to the current seed, and needed to return to it.
    pass

# sets the seed to the given value (or random), then iterates the given amount of times
def SetSeed(Seed=None, Iter=0):
    pass

# "pushes" the seed Iter times
def IterSeed(Iter=1):
    pass

class SeedBank:
    def __init__(self, seed=None, history=None, future=None):
        self.seed = seed if seed != None else random.randint(0, 2**32 - 1)
        self.history = history if history != None else [self.seed]
        self.future = future if future != None else []

    def get_seed(self):
        return self.seed

    def set_seed(self, seed=None, iter=0):
        if seed != None:
            self.seed = seed
            self.history.append(seed)
        for _ in range(iter):
            self.seed = random.randint(0, 2**32 - 1)
            self.history.append(self.seed)

    def next_seed(self):
        if self.future== None:
            self.set_seed(iter=1)


# Either runs the experiment with the given seed, OR it runs it with the current seed, which it then itterates
def RunExperiment(Funct, /, args=[], kwargs={}, Naming=None, seed=None, *, straightpass=False, KeepTime=True, Timename="time"):
    if straightpass: # you can also just wrap it in an additional list
        args = [args]
    elif args == None:
        args = []
    elif type(args) != list:
        args = [args]

    if kwargs == None:
        kwargs = dict()
    elif type(kwargs) != dict:
        kwargs = dict(kwargs)  # try to get a dict if possible.
        
    SeedSave = GetSeed()
    if seed != None:
        SetSeed(seed)
    if KeepTime:
        start_time = time.perf_counter()

    results = Funct(*args, **kwargs)

    if KeepTime:
        end_time = time.perf_counter()
    
    if Naming == None:          # no naming guide, leave as is
        pass
    elif type(Naming) == dict:  # we have a renaming for the input (doesnt need to be nice)
        results = {name : TryGet(results, k) for name,k in Naming}
    elif type(results) != dict: # unnice output, we have naming guide
        results = dict(rzip(Naming, results))
    else:                       # we have naming and imput is nice
        results = {name : TryGet(results, k) for name,k in rzip(Naming, results)}

    if KeepTime:
        if type(results) == dict:
            results[Timename] = end_time - start_time
        else:
            results = (*results, end_time - start_time)

    SetSeed(SeedSave)
    if seed == None:
        # if not fed a seed, iterate it
        IterSeed(1)
    
    return results


class VariableConfig:
    def __init__(self, config=None, unwrap=None):
        self.config = config if config != None else dict()
        self.unwrap = unwrap # defines how to "unwrap" the kwargs into an arg portion.
    
    def set_variable(self, name, values):
        self.config[name] = values

    def set_variables(self, names, values):
        for name, value in zip(names, values):
            self.set_variable(name, value)

    def extend_variable(self, name, values):
        if name in self.config:
            self.config[name].extend(values)
        else:
            self.set_variable(name, values)
    
    def get_variable(self, name):
        return self.config.get(name)

    def __str__(self):
        return str(self.config)

    def __iter__(self):
        for v in itertools.product(*self.config.values()):
            yield dict(zip(self.config.keys(), v))

    def __getitem__(self, key, idx=0):
        return self.config[key][idx]

    def __len__(self):
        return np.prod([len(v) for v in self.config.values()])

# runs X instances of Funct on the inputs args_c and kwarg_c, each with a succesive iteration of seed.
def MassExperiment(Funct, args_c, kwarg_c, repeats=1, Naming=None, seed=None, *, track = True, 
                   straightpass=False, KeepTime=True, Timename="time", Identifier=None):
    data = []
    for i in trange(repeats, desc='Experiment loop', leave=track):
        row = RunExperiment(Funct, args=args_c, kwargs=kwarg_c, Naming=Naming, seed=seed,
                            straightpass=straightpass, KeepTime=KeepTime, Timename=Timename)
        if type(Identifier) == str:
            row["id"] = Identifier + str(i)
        elif Identifier != None:
            row["id"] = Identifier(i)
        elif row.get("id") == None:
            row["id"] = i
        else:
            row["id"] = row["id"] + "_" + str(i)
        data.append(row)
    return data

# runs X instances of Funct on the inputs provided by Var_config, each with a succesive (or provided) iteration of seed.
def VariableExperiment(Funct, Var_config, repeats=1, Naming=None, seed=None, *, track = True, 
                       straightpass=False, KeepTime=True, Timename="time", Identifier=None):
    data = []
    """
    ran = trange(repeats, desc='Experiment loop', leave=track)
    for i in ran:
        row = RunExperiment(Funct, args=args_c, kwargs=kwarg_c, Naming=Naming, seed=seed,
                            straightpass=straightpass, KeepTime=KeepTime, Timename=Timename)
        if type(Identifier) == str:
            row["id"] = Identifier + str(i)
        elif Identifier != None:
            row["id"] = Identifier(i)
        else:
            row["id"] = i
        data.append(row)
    """
    return data



def ProcessResults(Results, Process):
    pass


if __name__ == "__main__":
    VCon = VariableConfig()
    VCon.set_variable("a", [1,2,3])
    VCon.set_variable("b", [4,5,6])
    print(list(VCon.__iter__()))
    it = 0
    for v in tqdm(VCon):
        #print(v)
        it += 1
        time.sleep(0.1)
    print (it, len(VCon))