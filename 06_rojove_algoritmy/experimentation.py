import time

def rzip(left,right):
    pass

def GetSeed(): # returns the value associated to the current seed, and needed to return to it.
    pass

# sets the seed to the given value (or random), then iterates the given amount of times
def SetSeed(Seed=None, Iter=0):
    pass

# "pushes" the seed Iter times
def IterSeed(Iter=1):
    pass

# Either runs the experiment with the given seed, OR it runs it with the current seed, which it then itterates
def RunExperiment(Funct, /, args=None, kwargs=None, Naming=None, seed=None, *, straightpass=False, KeepTime=True, Timename="time"):
    if straightpass:
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
    
    # 4 cazez
    if Naming != None and type(results) != dict:
        results = dict(zip(results,Naming))
    elif Naming != None:
        results = {name : results[k] for name,k in zip(Naming, results)}

    if KeepTime:
        if type(results) == dict:
            results[Timename] = end_time - start_time
        else:
            results = *results, end_time - start_time


    if seed == None:
        SetSeed(SeedSave, 1)
    
    return results
        
    


class VariableConfig:
    def __init__():
        pass

def MassExperiment(Funct, args_c, kwarg_c):
    pass

def VariableExperiment(Funct, Var_config, tuff):
    pass

Lia = [0,1,2,3]
Lib = [1,2,3,4]
Lic = [2,3,4,5]
print(list(zip(Lic,dict(zip(Lia,Lib)))))