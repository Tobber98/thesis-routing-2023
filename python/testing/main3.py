import numpy as np

from plot import Graph


class Routing:
    def __init__(self, zlim, ylim, xlim) -> None:
        self.zlim: int = zlim
        self.ylim: int = ylim
        self.xlim: int = xlim

        self.g           = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.h           = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.f           = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.accessible  = np.ones(self.zlim * self.ylim * self.xlim, dtype=int)
        self.predecessor = np.ones(self.zlim * self.ylim * self.xlim, dtype=int) * -1

    def reset_mesh(self, reset_accessible = False):
        self.g = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.h = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.f = np.ones(self.zlim * self.ylim * self.xlim) * np.inf
        self.predecessor = np.ones(self.zlim * self.ylim * self.xlim, dtype=int) * -1
        if reset_accessible:
            self.accessible = np.ones(self.zlim * self.ylim * self.xlim, dtype=int)

    def index_unravel(self, i):
        # return coordinates z, y, x
        return (i // (self.ylim * self.xlim), 
                i // self.xlim % self.ylim, 
                i % self.xlim)

    def index_ravel(self, z, y, x):
        return z * (self.ylim * self.xlim) + y * self.xlim + x
    
    def get_successors(self, q, goal):
        z, y, x = self.index_unravel(q)
        zoffset = self.ylim * self.xlim
        return np.array(
            ([q - zoffset]   if (z > 0               and self.accessible[q - zoffset]  ) or q - zoffset   == goal else []) +
            ([q + zoffset]   if (z < self.zlim - 1   and self.accessible[q + zoffset]  ) or q + zoffset   == goal else []) +
            ([q - self.xlim] if (y > 0               and self.accessible[q - self.xlim]) or q - self.xlim == goal else []) +
            ([q + self.xlim] if (y < self.ylim - 1   and self.accessible[q + self.xlim]) or q + self.xlim == goal else []) +
            ([q - 1]         if (x > 0               and self.accessible[q - 1]        ) or q - 1         == goal else []) +
            ([q + 1]         if (x < self.xlim - 1   and self.accessible[q + 1]        ) or q + 1         == goal else []))

    def absolute_distance(self, viable, goal):
        return np.sum(abs(self.index_unravel(viable) - np.array(self.index_unravel(goal))[:,None]), 0)

    def a_star(self, start, goal):
        self.g[start] = 0
        self.f[start] = 0

        open = [start]
        closed = []
        
        while open:
            q = open.pop(np.argmin(list(map(lambda x: self.f[x], open))))
            closed.append(q)

            succesors = self.get_successors(q, goal)
            if goal in succesors:
                self.predecessor[goal] = q
                break
            
            viable = np.intersect1d(np.argwhere(self.g > self.g[q] + 1), np.setdiff1d(succesors, closed))
            self.g[viable] = self.g[q] + 1
            # self.h[viable] = self.absolute_distance(viable, goal)
            # self.f[viable] = self.g[viable] + self.h[viable]
            self.f[viable] = self.g[viable] + self.absolute_distance(viable, goal)
            self.predecessor[viable] = q
            open.extend(viable)
            open = list(set(open))

        path = []
        current = goal
        while current != start:
            self.accessible[current] = 0
            path.append(self.index_unravel(current))
            current = self.predecessor[current]
        path.append(self.index_unravel(start))
        return path

    def run(self, starting_points, goals):
        paths = []
        self.plot_init()
        self.accessible[starting_points] = 0
        self.accessible[goals] = 0
        for starting_point, goal in zip(starting_points, goals):
            paths.append(self.a_star(starting_point, goal))
            self.reset_mesh()
        return paths
            

if __name__ == "__main__":
    grid = Routing(5, 8, 8)

    starts = []
    goals = []

    path = grid.a_star(grid.index_ravel(1, 3, 2), grid.index_ravel(2, 5, 6))
    print(path)
    graph = Graph(5, 8, 8)
    graph.plot_init()
    graph.plot_path(path, "red")
    graph.show()