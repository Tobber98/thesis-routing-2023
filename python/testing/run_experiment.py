import enum
from platform import node
from src.routing.routing import Routing
from src.plotter.plotter import PlotlyGraph
from src.utils import data_reader, nodes_to_indices

from pathlib import Path

import cProfile
import pstats

import pandas as pd

ZDIM = 7
DATA = "../data"
FILENAME = "test_data.json"

MESHES_PATHS = ("x20y20", "x30y30", 
        "x40y40", "x50y50", 
        "x60y60", "x70y70", 
        "x80y80", "x90y90")

def main():
    r = Routing()

    for mesh_path in MESHES_PATHS[::-1]:
        print(mesh_path)
        d = data_reader((Path(DATA) / mesh_path / FILENAME))
        x, y = map(int, d["dims"])
        r.set_mesh_dims(ZDIM, y, x)

        for i in range(10, 91):
            print(" ", i)
            a = nodes_to_indices(d["indices"], d["pathlists"][f"N{i}"])
            for j, b in enumerate(a):
                print("   ", j)
                q = r.run3(list(b))
                # gather data
                q = list(filter(None, q))
                r.reset_mesh()
            break
        break



    # d = data_reader(X20Y20)
    # a = nodes_to_indices(d["indices"], d["pathlists"]["N20"])
    # x, y = map(int, d["dims"])
    # r.set_mesh_dims(ZDIM, y, x)

    # for i, b in enumerate(a):
    #     print(i)
    #     q = r.run3(list(b))
    #     q = list(filter(None, q))
    #     r.reset_mesh()

with cProfile.Profile() as pr:
    main()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
stats.dump_stats(filename="test.prof")


# First read data from file(s)
# Use node index key to turn node points into locations