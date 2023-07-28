# import random
# from src.routing.routing import Routing
# from src.plotter.plotter import PlotlyGraph as Graph

# import cProfile
# import pstats


# from pathlib import Path

# def main():
#     r = Routing(7, 100, 100)
#     g = Graph(7, 100, 100)

#     random_list = random.sample(range(0, 7 * 100 * 100), 80)
#     start = random_list[: len(random_list) // 2]
#     end = random_list[len(random_list) // 2 :]

#     paths = r.run(start, end)

#     g.plot_paths(paths)
#     g.show()

# with cProfile.Profile() as pr:
#     main()

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# stats.dump_stats(filename="test.prof")

# import random
# import numpy as np
# from math import factorial as f

# size = 60
# draws = 20

# total = []
# x = list(range(size))
# for i in range(100000):
#     a = set()
#     for _ in range(draws):
#         r = random.randint(0, size - 1)
#         a.add(x[r])
#     total.append(len(a))

# b = np.array(total)
# c = b // draws
# print(sum(c) / 100000 * 100)

# x = 0
# for i in range(1, 10):
#     x = x + (i / 100)
# print((1-x) * 100)

# t = lambda n, c: (n**c - f(n) / f(n - c)) / n**c
# t = lambda n, c: (f(n) / f(n - c)) / n**c
# print((t(size, draws)) * 100)
# n = 10
# c = 3

# x = 0
# for i in range(1, 3):
#     x += i / 11
# print((1-x) * 100)

# print(n / n**3)
# from pathlib import Path
# import json

# MESHSIZES = list(range(20, 101, 10))
# data_paths = [Path(f"../data/x{i}y{i}/test_data.json") for i in MESHSIZES]

# j = 0
# j2 = 0
# for m, data_path in zip(MESHSIZES, data_paths):
#     with open(data_path, "r") as fd:
#         data = json.load(fd)
#     indices = data["indices"]
#     pls = data["pathlists"]
#     for pl in pls:
#         for sample, a in enumerate(pls[pl]):
#             j += 1
#             g = False
#             for i, b in enumerate(a):
#                 res = abs(indices[b[1]] - indices[b[0]])
#                 if res == 1 or res == m:
#                     g = True
#                     print(m, pl, b, i, sample)
#             if g:
#                 j2 += 1

# print(j, j2, j2 / j)


# import numpy as np
from itertools import permutations, product

pairs = ["a", "b", "c"]
perms = list(permutations(pairs))

# s2 = list(product(perms, repeat=2))
# s3 = list(product(perms, repeat=3))
s4 = list(product(perms, repeat=4))
# s5 = list(product(perms, repeat=5))
# s6 = list(product(perms, repeat=6))
# s7 = list(product(perms, repeat=7))

# print((s2[:, 0] == s2[:, 1]).all(-1))
unique = 0
double = 0
for s in s4:
    if len(set(s)) == len(s):
        unique += 1
    else:
        double += 1
print(unique, double)
print(1 - unique / (unique + double))
# # print(s3)

from math import factorial as f
def f1(l, s):
    return 1 - (f(f(l)) / f(f(l) - s)) / (f(l) ** s)

def f2(l, s):
    return 

print(f1(3, 4))
print(f2(3, 4))