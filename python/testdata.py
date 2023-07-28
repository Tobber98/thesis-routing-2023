from src.routing.routing import Routing
from src.plotter.plotter import PlotlyGraph as Graph

import pandas as pd



# For each x by y mesh create a dictionary containing as key the xy and value mesh, 
# key with pathlists, values of pathlists is a dictionary.
# pathlist dict contains keys for all Ns with as name Nxx and value list of all 20 pathlists.



def read_data():
    d = dict()
    with open("../data/original/x20y20/C100.csv") as fd:
        text = fd.read()
    vals = ",".join(text.strip().split("\n")).split(",")
    for i, val in enumerate(vals):
        d[val] = i
    
    l = []
    with open("../data/original/x20y20/C100/N10/N10_0.csv") as fd:
        text = fd.read()
    vals = ",".join(text.strip().split("\n")).split(",")
    while(vals):
        n = vals.pop(0)
        start = vals.pop(0)
        end = vals.pop(0)
        l.append((d[start], d[end]))
    return l


def read_data_order(path):
    import json
    d = dict()
    with open(path) as fd:
        data = json.load(fd)

    a = data["netlists"]["N20"][0]
    mapping = data["shapes"]["(12, 12, 12)"][0]
    b = list(map(lambda x: [mapping[x[0]], mapping[x[1]]], a))
    return b, mapping


def runtest(l):
    r = Routing(7, 20, 20)
    g = Graph(7, 20, 20)

    paths = r.run2(l)

    g.plot_paths(paths)
    g.show()

def runtest_order(l, m):
    def index_unravel(i):
        y, x = 12, 12
        return (i // (y * x), i // x % y, i % x)
    r = Routing(12, 12, 12)
    g = Graph(12, 12, 12)

    paths = r.run2(l)


    data_coords = map(index_unravel, m)
    g.plot_point(zip(*data_coords), 100)
    g.plot_paths(paths)
    g.show()

if __name__ == "__main__":
    l = read_data()
    runtest(l)

    l, m = read_data_order("../data/orderrouting_sides/test_data.json")
    runtest_order(l, m)

    l, m = read_data_order("../data/orderrouting/test_data.json")
    runtest_order(l, m)
    # 5-vvz-23