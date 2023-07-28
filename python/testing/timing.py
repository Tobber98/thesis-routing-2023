import time
from typing import Callable
import numpy as np
import heapq

from copy import copy


def time_func(func, n: Callable):
    start_timer = time.perf_counter_ns()
    for _ in range(n):
        a = func()
    end_timer = time.perf_counter_ns()
    print(f"Function[{func.__name__}]: {(end_timer - start_timer) / n}")


def test():
    a = 1 + 3
    b = a + 2


time_func(test, 10000) 

"""
x = 5
y = 4
z = 3

r = Routing(z, y, x)

i = 7
coord = (0, 1, 2)

start_own_unravel = time.perf_counter_ns()
for b in range(10000000):
    r.index_unravel(i)
end_own_unravel = time.perf_counter_ns()

start_np_unravel = time.perf_counter_ns()
for b in range(10000000):
    np.unravel_index(i, (z, y, x))
end_np_unravel = time.perf_counter_ns()

start_own_ravel = time.perf_counter_ns()
for b in range(10000000):
    r.index_ravel(*coord)
end_own_ravel = time.perf_counter_ns()

start_np_ravel = time.perf_counter_ns()
for b in range(10000000):
    np.ravel_multi_index(coord, (z, y, x))
end_np_ravel = time.perf_counter_ns()

print(f"Own Unravel: {(end_own_unravel - start_own_unravel) / 10000000}\n \
    NP Unravel: {(end_np_unravel - start_np_unravel) / 10000000}\n \
    Own Ravel: {(end_own_ravel - start_own_ravel) / 10000000}\n \
    NP Ravel: {(end_np_ravel - start_np_ravel) / 10000000}\n ")
"""


"""
start_np_succesor = time.perf_counter_ns()
for b in range(1000000):
    np.array(
    (
    ([2]) +
    ([10222]) +
    ([]) +
    ([123]) +
    ([999]) +
    ([])))
end_np_succesor = time.perf_counter_ns()

start_tuple_successor = time.perf_counter_ns()
for b in range(1000000): 
    (
    ([2]) +
    ([10222]) +
    ([]) +
    ([123]) +
    ([999]) +
    ([]))
end_tuple_successor = time.perf_counter_ns()

print(f"NP succesor : {(end_np_succesor - start_np_succesor) / 10000000}\n \
    Own succesor: {(end_tuple_successor - start_tuple_successor) / 10000000}\n")
"""

# l = list(range(1, 100000))

# start_slice_timer = time.perf_counter_ns()
# for i in range(10000):
#     l2 = l[1+i*3 : 3+i*3]
#     # l2 = l[1:3]
#     # del l[:3]
# del l
# stop_slice_timer = time.perf_counter_ns()

# l = list(range(1, 100000))
# start_pop_timer = time.perf_counter_ns()
# for i in range(10000):
#     a = l.pop(0)
#     b = l.pop(0)
#     c = l.pop(0)
#     d = [b, c]
# stop_pop_timer = time.perf_counter_ns()

# print(f"Slice: {(stop_slice_timer - start_slice_timer) / 10000000}\n\
# Pop: {(stop_pop_timer - start_pop_timer) / 10000000}\n")

"""
l1 = [2, 4, 3, 1, 10, 15]
l2 = [2, 4, 3, 1, 10, 15] + list(range(20, 1050))

tc = [1, 7, 3, 10000]
start_list_timer = time.perf_counter_ns()
for i in range(10000):
    l1c = l1.copy()
    l2c = l2.copy()

    l1c.extend(tc)
    l1c = list(set(l1c))
    l2c.extend(tc)
    l2c = list(set(l2c))

    l1c.pop(np.argmin(l1c))
    l2c.pop(np.argmin(l2c))
stop_list_timer = time.perf_counter_ns()

heapq.heapify(l1)
heapq.heapify(l2)

start_heap_timer = time.perf_counter_ns()
for i in range(10000):
    h1c = copy(l1)
    h2c = copy(l2)

    for a in tc:
        if a not in h1c:
            heapq.heappush(h1c, a)

    for a in tc:
        if a not in h2c:
            heapq.heappush(h2c, a)

    heapq.heappop(h1c)
    heapq.heappop(h2c)

stop_heap_timer = time.perf_counter_ns()
print(f"List: {(stop_list_timer - start_list_timer) / 10000000}\n\
Heap: {(stop_heap_timer - start_heap_timer) / 10000000}\n")
"""

l = [1, 4, 5, 76, 5, 3, 2, 6, 34, 4256, 34, 23, 1, 234, 423, 5324 , 100]
arr = np.array(l)

start_list_timer = time.perf_counter_ns()
for i in range(10000000):
    a = l[2]
    b = l[6]
stop_list_timer = time.perf_counter_ns()

start_np_timer = time.perf_counter_ns()
for i in range(10000000):
    a, b = arr[[2,6]]
stop_np_timer = time.perf_counter_ns()

print(f"List: {(stop_list_timer - start_list_timer) / 10000000}\n\
Numpy: {(stop_np_timer - start_np_timer) / 10000000}\n")