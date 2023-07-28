import heapq
from itertools import chain
# Look into heapdict instead heapq
import numpy as np

class Routing:
    def __init__(self, zlim: int = 10, ylim: int = 10, xlim: int = 10) -> None:
        self.set_mesh_dims(zlim, ylim, xlim)

    def set_mesh_dims(self, zlim: int, ylim: int, xlim: int):
        self.zlim: int = zlim
        self.ylim: int = ylim
        self.xlim: int = xlim

        self.g           = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.f           = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.accessible  = np.ones(self.zlim * self.ylim * self.xlim, dtype=bool)
        self.predecessor = np.ones(self.zlim * self.ylim * self.xlim, dtype=int) * -1
        self.neighbours  = self.set_neighbours()

    def reset_mesh(self, reset_accessible: bool = False):
        self.g = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.f = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.predecessor = np.ones(self.zlim * self.ylim * self.xlim, dtype=int) * -1
        if reset_accessible:
            self.accessible = np.ones(self.zlim * self.ylim * self.xlim, dtype=int)

    def set_neighbours(self):
        neighbours = np.empty(self.zlim * self.ylim * self.xlim, dtype=object)
        for i in range(len(neighbours)):
            neighbours[i] = self.get_neighbours(i)
        return neighbours

    def get_neighbours(self, i: int):
        z, y, x = self.index_unravel(i)
        z_offset = self.ylim * self.xlim
        return np.array(
            ([i - z_offset]   if (z > 0             ) else []) +
            ([i + z_offset]   if (z < self.zlim - 1 ) else []) +
            ([i - self.xlim]  if (y > 0             ) else []) +
            ([i + self.xlim]  if (y < self.ylim - 1 ) else []) +
            ([i - 1]          if (x > 0             ) else []) +
            ([i + 1]          if (x < self.xlim - 1 ) else []))

    def index_unravel(self, i: int):
        # return np.unravel_index(i, (self.zlim, self.ylim, self.xlim))
        return (i // (self.ylim * self.xlim), 
                i // self.xlim % self.ylim, 
                i % self.xlim)

    def index_ravel(self, z: int, y: int, x: int):
        # return np.ravel_multi_index(coordinates, (self.zlim, self.ylim, self.xlim))
        return z * (self.ylim * self.xlim) + y * self.xlim + x

    def get_successors(self, q: int, goal: int):
        z, y, x = self.index_unravel(q)
        z_offset = self.ylim * self.xlim

        return (
            ([q - z_offset]   if z > 0              
                and (self.accessible[q - z_offset] 
                or q - z_offset  == goal) else []) +
            ([q + z_offset]   if z < self.zlim - 1  
                and (self.accessible[q + z_offset] 
                or q + z_offset  == goal) else []) +
            ([q - self.xlim] if y > 0               
                and (self.accessible[q - self.xlim]
                or q - self.xlim == goal) else []) +
            ([q + self.xlim] if y < self.ylim - 1   
                and (self.accessible[q + self.xlim]
                or q + self.xlim == goal) else []) +
            ([q - 1]         if x > 0               
                and (self.accessible[q - 1]        
                or q - 1        == goal) else []) +
            ([q + 1]         if x < self.xlim - 1   
                and (self.accessible[q + 1]        
                or q + 1         == goal) else []))
    
    def get_successors2(self, q: int, goal: int):
        bc = self.neighbours[q]
        if goal in bc:
            return goal
        return bc[self.accessible[bc]]

    def absolute_distance(self, viable: int, goal: int):
        return sum(abs(self.index_unravel(viable) - 
                np.array(self.index_unravel(goal))[:,None]), 0)

    def check_goal_surrounded(self, goal):
        pass

    def a_star(self, start, goal):
        self.g[start] = 0
        # self.f[start] = 0

        open_heap = []
        open_set = {start}
        heapq.heappush(open_heap, (0, start))
        closed = set()
        print("\n\n")
        while open_heap:
            q = heapq.heappop(open_heap)[1]
            open_set.remove(q)
            closed.add(q)

            succesors = self.get_successors(q, goal)
            print(succesors)
            if goal in succesors:
                self.predecessor[goal] = q
                break

            unclosed = np.array(tuple(set(succesors) - closed), dtype=int)
            viable = unclosed[self.g[unclosed] > self.g[q]]
            self.g[viable] = self.g[q] + 1
            sf = self.g[viable] + self.absolute_distance(viable, goal)
            self.predecessor[viable] = q
            for v, f in zip(viable, sf):
                if v not in open_set:
                    heapq.heappush(open_heap, (f, v))
                    open_set.add(v)

        path = []
        current = goal
        while current != start:
            if current == -1:
                return [] 
            self.accessible[current] = 0
            path.append(self.index_unravel(current))
            current = self.predecessor[current]
        path.append(self.index_unravel(start))
        return path


    def run(self, starting_points, goals):
        paths = []
        self.accessible[starting_points] = 0
        self.accessible[goals] = 0
        for starting_point, goal in zip(starting_points, goals):
            paths.append(self.a_star(starting_point, goal))
            self.reset_mesh()
        return paths

    # make run2 or run3 main later    
    def run2(self, points):
        paths = []
        self.accessible[points] = 0
        for starting_point, goal in points:
            paths.append(self.a_star(starting_point, goal))
            self.reset_mesh()
        return paths

    def run3(self, *points):
        paths = []
        points = points[0] if len(points) == 1 else zip(*points)
        self.accessible[list(chain(*points))] = 0
        print(points)
        for starting_point, goal in points:
            paths.append(self.a_star(starting_point, goal))
            self.reset_mesh()
        print(paths)
        return paths