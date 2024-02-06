def Loss(circle_points, edges, r, Ns):
    loss = 0
    for p in circle_points:
        try:
            loss += edges[p[0]][p[1]]
        except IndexError:
            continue
    k = 0.7
    rt = 500
    if r > rt: 
        fr = 1
    else:
        fr = k * r / rt
    loss = loss / Ns * fr
    return loss

def GA(edges, MaxIt = 5, nPop = 10, Ns = 200, pc = 0.8, pm = 0.2, mu = 0.02, beta = 8):
    import numpy as np
    import math
    import random

    progress = 0
    nm  = round(pm * nPop)
    nc = 2 * round(pc * nPop / 2)

    print("Initialization ...")

    edge_points = []
    for i in range(len(edges)):
        for j in range(len(edges[0])):
            if edges[i][j] == 255:
                edge_points.append((i, j))
        
    class Choromosome():
        def __init__(self, arr):
            self.string = arr
            self.r, self.c = find_circle(edge_points[arr[0]], edge_points[arr[1]], edge_points[arr[2]])
            self.circle_points = []
            # Ns = 200
            for i in range(Ns):
                x = int(self.c[0] + self.r * math.sin(2 * math.pi * i / Ns))
                y = int(self.c[1] + self.r * math.cos(2 * math.pi * i / Ns))
                self.circle_points.append((x, y))
            self.cost = Loss(self.circle_points, edges, self.r, Ns)
    
    mating_pool = []
    
    for i in range(nPop):
        arr = np.random.randint(len(edge_points), size=3)
        progressbar(100, progress/nPop)
        mating_pool.append(Choromosome(arr))
        progress += 1
        progressbar(100, progress/nPop)

    mating_pool = sorted(mating_pool, key=lambda x: x.cost, reverse = True)

    best_sol = [mating_pool[0]] 
    average_cost = [np.mean([ch.cost for ch in mating_pool])]
    # best_cost = [mating_pool[0].cost]
    worst_cost = mating_pool[-1].cost
    print(f"\nBest Solution: {best_sol[-1].string}\nBest Cost: {best_sol[-1].cost}\nAverage Cost: {average_cost[-1]}")

    for it in range(MaxIt):
        next_pool = []
        print("\n\nStarting Iteration Number", it + 1, "...")
        progress = 0
        progressbar(100, progress/(nm + nc))
        # P = exp(-1 * beta * [1, 2, 3])
        P = [chromosome.cost for chromosome in mating_pool]
        sum_p = sum(P)
        P = [x / sum_p for x in P]

        # Crossover
        indices = []
        for k in range(nc):
            ch = np.random.choice(nPop, p=P)
            while ch in indices:
                ch = np.random.choice(nPop, p=P)
            indices.append(ch)

        for k in range(int(nc / 2)):
            arr1 = mating_pool[indices[2 * k]].string 
            arr2 = mating_pool[indices[2 * k + 1]].string

            child1 = np.concatenate((arr1[:1], arr2[1:]), axis=0)
            
            child2 = np.concatenate((arr2[:1], arr1[1:]), axis=0)

            next_pool.append(Choromosome(child1))
            progress += 1
            progressbar(100, progress/(nm + nc))
            next_pool.append(Choromosome(child2))
            progress += 1
            progressbar(100, progress/(nm + nc))
            
        for k in range(nm):
            ch = np.random.choice(nPop, p=P)
            rnd = random.randint(0, len(edge_points)-1)
            child = mating_pool[ch].string
            ind = np.random.randint(3)
            child[ind] = rnd
            next_pool.append(Choromosome(child))
            progress += 1
            progressbar(100, progress/(nm + nc))

        mating_pool = sorted(mating_pool + next_pool, key=lambda x: x.cost, reverse = True)[:nPop]
        worst_cost = mating_pool[-1].cost
        average_cost.append(np.mean([ch.cost for ch in mating_pool]))
        best_sol.append(mating_pool[0])
        print(f"\nBest Solution: {best_sol[-1].string}\nBest Cost: {best_sol[-1].cost}\nAverage Cost: {average_cost[-1]}")
        

    return {"best_sol": best_sol,
            "average_cost": average_cost,
            "Population": mating_pool}

import sys
import math
def progressbar(length, progress):

    bar = "#" * int(length * progress)
    spaces = " " * (length - len(bar))
    sys.stdout.write(f"\r[{bar}{spaces}] {int(progress * 100)}%")
    sys.stdout.flush()

def find_circle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculating the midpoints of two line segments
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    x_perp = (x2 - x1) / 2
    y_perp = (y2 - y1) / 2

    # Checking if the line segments are parallel
    if x2 - x1 == 0:
        slope1 = None
    else:
        slope1 = (y2 - y1) / (x2 - x1)

    if x3 - x2 == 0:
        slope2 = None
    else:
        slope2 = (y3 - y2) / (x3 - x2)

    # Calculating the center coordinates of the circle
    if slope1 is None:
        center_x = x_mid
        try:
            center_y = slope2 * (center_x - x3) + y3
        except TypeError:
            center_y = 0
    elif slope2 is None:
        center_x = x3
        center_y = slope1 * (center_x - x_mid) + y_mid
    else:
        try:
            center_x = (slope1 * slope2 * (y3 - y1) + slope1 * (x2 + x3) - slope2 * (x1 + x2)) / (2 * (slope1 - slope2))
        except ZeroDivisionError:
            center_x = 10
        if slope1 != 0:
            center_y = -(1 / slope1) * (center_x - x_mid) + y_mid
        else:
            try:
                center_y = -(1 / slope2) * (center_x - x3) + y3
            except ZeroDivisionError:
                center_y = 1

    # Calculating the radius of the circle
    r = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

    return r, (center_x, center_y)

    