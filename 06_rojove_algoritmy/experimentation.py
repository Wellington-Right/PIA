import time
import itertools
import random
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import copy

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

namemap = {list : "List"}

def Nicerep(arg, push=0):
    global namemap
    sep = (push-1)*"│ " + "├─ " if push>0 else ""
    blocks = []
    if   type(arg) == list:
        blocks = [sep + str(k) for k in arg]
    elif type(arg) == VariableConfig:
        blocks = [sep + str(k) + " : " +  str(arg.config[k]) for k in arg.config]
    elif type(arg) == ConfigChain:
        blocks = [sep + f"{namemap[type(arg)]} block {idx:02d}:\n" + Nicerep(item, push+1) 
                  for idx, item in enumerate(arg.chain)]
    elif type(arg) == CartesianConfig:
        blocks = [sep + f"{namemap[type(arg)]} block {idx:02d}:\n" + Nicerep(item, push+1) 
                  for idx, item in enumerate(arg.chain)]
    else:
        raise NotImplementedError()
    return "\n".join(blocks)

# the implementation below is not designed with, speed in mind, its just a QOL
"""
stores and interprets a variable configuration inside self.config.
its depicted as a dictionary of lists which hold the valid members to the given configuration
desired methods:
/ for all
    __init__        
    validate        
    validate_idx    
    key_iter        
    value_iter      
    __iter__        
    __getitem__     
    __len__         
    __iadd__        
    __add__         
    __str__         
    set_variable    
    set_variables   
    extend_variable 
    get_variable    
"""
class VariableConfig: 
    def __init__(self, config=None, constrains=None):
        """ 
        initiates the variable Config construct
        assume that the config parameter is properly formatted
        """
        self.config = config if config != None else dict()
        self.stable = False
        self.Lenght = 0
        self.constrains = []
        if constrains != None:
            self.constrains = list(constrains) if hasattr(constrains, '__iter__') else [constrains]

    def set_variable(self, name, values):
        self.config[name] = values
        self.stable = False

    def set_variables(self, names, values):
        self.stable = False
        for name, value in zip(names, values):
            self.set_variable(name, value)

    def extend_variable(self, name, values):
        self.stable = False
        if name in self.config:
            self.config[name].extend(values)
        else:
            self.set_variable(name, values)
    
    def get_variable(self, name):
        return self.config.get(name)

    def add_constraint(self, constraint):
        self.stable = False
        self.constrains.append(constraint)

    def validate(self, varset):
        for check in self.constrains:
            if not check(varset):
                return False
        return True
    
    def validate_idx(self, idx):
        return (not self.constrains) or self.validate(self[idx])

    # same as above but inplace
    def __iadd__(self, other):
        # no need to be fancy about it
        return ConfigChain(self, other)

    # addition should function as simple chaining.
    def __add__(self, other):
        # no need to be fancy about it
        return ConfigChain(self, other)

    # extends the variable configuration by the specified item TODO: generalise maybe
    def __imul__(self, other):
        data = dict()
        if   type(other) in [VariableConfig]:   # stuff we want to add as an element
            data = other.config
        elif type(other) in [dict]:             # stuff we want to encapsulate
            data = other
        else:                                   # what to do with unrecognizable stuff
            raise NotImplementedError()         # TODO: figure how to handle the annoying cases (probably just dont handle them)
        for key in data:
            self.extend_variable(key, data[key])
        return self

    # extends the variable configurations by the specified items configurations
    def __mul__(self, other):
        res = copy.copy(self)
        res *= other
        return res

    # should also somehow represent the existence of constraints TODO: pretify
    def __str__(self):
        return Nicerep(self)

    # iterates over indexes to configurations
    def key_iter(self):
        idx_ranges = [range(len(vals)) for vals in self.config.values()]
        for idx in itertools.product(*idx_ranges):
            if not self.validate_idx(idx):
                continue
            yield idx

    # iterates over the configurations
    def value_iter(self):
        for v in itertools.product(*self.config.values()):
            choice = dict(zip(self.config.keys(), v))
            if not self.validate(choice):
                continue
            yield choice

    # maybe iterate over key, idx pairs instead?
    __iter__ = value_iter

    # Ignores Constraints, REMEMBER THIS
    def __getitem__(self, idx):
        choice = {}
        for key, i in zip(self.config.keys(), idx):
            choice[key] = self.config[key][i]
        return choice

    # you shouldnt modify this thing too much between calls for lenght so this is probably fine.
    def __len__(self):
        if self.stable:
            return self.Lenght
        self.stable = True
        self.Lenght = 0
        if self.constrains == None or self.constrains == []:
            self.Lenght = np.prod([len(v) for v in self.config.values()])
        else:
            # this is basically the worst way to do it, but it also works
            for _ in self.__iter__():
                self.Lenght += 1
        return self.Lenght
namemap[VariableConfig] = "ConVar"
class ConfigChain:
    def __init__(self, *args, constrains=None):
        self.chain = list(args)
        self.stable = False
        self.Lenght = 0
        self.constrains = []
        if constrains != None:
            self.constrains = list(constrains) if hasattr(constrains, '__iter__') else [constrains]

    def extend_by(self, *args):
        self.chain.extend(args)
        if self.stable:
            for arg in args:
                for item in arg.value_iter():
                    self.Lenght += self.validate(item)

    def add_constraint(self, constraint):
        self.stable = False
        self.constrains.append(constraint)

    def validate(self, varset):
        for check in self.constrains:
            if not check(varset):
                return False
        return True
    
    def validate_idx(self, idx):
        return (not self.constrains) or self.validate(self[idx])

    def __iadd__(self, other):
        if   type(other) in [ConfigChain]:      # stuff we want to extend from the back
            self.chain.extend(other.chain)
        elif type(other) in [VariableConfig]:   # stuff we want to add as an element
            self.chain.append(other)
        elif type(other) in [dict]:             # stuff we want to encapsulate
            self.chain.append([other])
        else:                                   # what to do with unrecognizable stuff
            assert(hasattr(other, '__iter__'))
            self.chain.append(other)
        return self

    def __add__(self, other):
        new = ConfigChain(self.chain)
        new += other
        return new

    # extends the variable configuration by the specified item TODO: generalise maybe
    def __imul__(self, other):
        return self
        data = dict()
        if   type(other) in [VariableConfig]:   # stuff we want to add as an element
            data = other.config
        elif type(other) in [dict]:             # stuff we want to encapsulate
            data = other
        else:                                   # what to do with unrecognizable stuff
            raise NotImplementedError()         # TODO: figure how to handle the annoying cases (probably just dont handle them)
        for key in data:
            self.extend_variable(key, data[key])

    # extends the variable configurations by the specified items configurations
    def __mul__(self, other):
        res = copy.copy(self)
        res *= other
        return res

    def __str__(self):
        return Nicerep(self)
    
    # iterates over keys TODO
    def key_iter(self):
        for i, r in enumerate(self.chain):
            indices = r.key_iter() if hasattr(r, 'key_iter') else range(len(r))
            for j in indices:
                yield (i, j)

    # iterates over the values TODO
    def value_iter(self):
        for link in self.chain:
            for config in link:
                yield config

    __iter__ = key_iter

    def __getitem__(self, idx):
        i ,*j = idx
        if len(j) == 1:
            j = j[0]
        #print (j, type(j))
        return self.chain[i][j]

    def __len__(self):
        return np.sum([len(link) for link in self.chain])
namemap[ConfigChain] = "VarSum"
class CartesianConfig:
    pass
namemap[CartesianConfig] = "VarProd"
# runs X instances of Funct on the inputs args_c and kwarg_c, each with a succesive iteration of seed.
def MassExperiment(Funct, args_c, kwarg_c, repeats=1, Naming=None, seed=None, *, track = True, 
                   straightpass=False, KeepTime=True, Timename="time", Identifier=None):
    data = []
    for i in trange(repeats, desc='Experiment loop', disable = track):
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
    BCon = VariableConfig()
    BCon.set_variable("a", [4,5])
    BCon.set_variable("b", [7,8,9])
    VCon += BCon
    VCon += [{"a" : None, "b" : None}, {"a" : None, "b" : None}]
    print(list(VCon.__iter__()))
    it = 0
    print(VCon)
    print(ConfigChain(VCon, VCon))
    for k in tqdm(VCon):
        tqdm.write(str(k) + " - " + str(VCon[k]))
        it += 1
        time.sleep(0.1)
    print((0,0,0), VCon[(0,0,0)])
    print (it, len(VCon))