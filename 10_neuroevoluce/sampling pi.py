import random
import numpy as np
from tqdm import *
import time

def simulateA(n, seed=None, pbar=None):
    if seed is None:
        random.seed()
    else:
        random.seed(seed)
    countA = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if np.square(x) + np.square(y) <= 1:
            countA += 1
        if pbar:
            pbar.update(1)
    return countA

def simulateB(n, seed=None, pbar=None):
    if seed is None:
        random.seed()
    else:
        random.seed(seed)
    sumB = 0
    for _ in range(n):
        sumB += np.sqrt(1- np.square(random.random()))
        if pbar:
            pbar.update(1)
    return sumB

def TestBarA(n=5,id="A",*agrs, **kwargs):
    tick = n / 100
    with tqdm(total=100,desc=id, *agrs, **kwargs) as pbar:
        for i in range(100):
            time.sleep(tick)
            pbar.update(1)
def TestBarB(n=5,id="B",*agrs, **kwargs):
    tick = n / 100
    for i in trange(100, desc=id, *agrs, **kwargs):
        time.sleep(tick)

TestBarA(10)

for i in trange(5, desc="A"):
    for j in trange(5, desc="B", leave=False):
        time.sleep(0.5)
    time.sleep(1.5)

print("P")

for i in trange(10, desc="A"):
    for j in trange(10, desc="B"+str(i)):
        time.sleep(0.1)

print("P")

TestBarA(1)
TestBarB(1)
for i in trange(5, desc="C"):
    TestBarA(i, id="A"+str(i),leave=False,colour="green")
    TestBarB(i, id="B"+str(i),leave=False)

exit()
data = []

for k in trange(9, position=0, leave=True):
    n = 10 ** k
    with tqdm(total=n, position=1, leave=False, desc="Loop-A") as pbarA, tqdm(total=n, position=2, leave=False, desc="Loop-B") as pbarB:
        CA = simulateA(n, 19495400, pbarA)
        pbarA.refresh()
        CB = simulateB(n, 19495400, pbarB)
    data.append((n, CA, CB))
    tqdm.write(f"n = {n:9} | Ea*4 = {(CA/n*4):2.7f} | Eb*4 = {(CB/n*4):2.7f} | |Ea*4-pi| = {abs(CA/n*4 - np.pi):2.7f} | |Eb*4-pi| = {abs(CB/n*4 - np.pi):2.7f}")
    
for n, CA, CB in data:
    tqdm.write(f"n = {n:9} | Ea*4 = {(CA/n*4):2.7f} | Eb*4 = {(CB/n*4):2.7f} | |Ea*4-pi| = {abs(CA/n*4 - np.pi):2.7f} | |Eb*4-pi| = {abs(CB/n*4 - np.pi):2.7f}")