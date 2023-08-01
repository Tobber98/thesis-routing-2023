import os
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np


def i_to_coord(index, shape):
    result = []
    base = np.prod(shape)
    for limit in shape:
        base //= limit
        result.append(index // base % limit)
    return result


def set_eligible_vertices(shape, min=None, max=None):
    eligible = []
    min = min if min else len(shape) + 2
    max = max if max else len(shape) * 2 - 1
    for i in range(np.prod(shape)):
        coord = np.array(i_to_coord(i, shape))
        if sum(coord != 0) + sum(coord != np.array(shape) - 1) >= min and sum(coord != 0) + sum(coord != np.array(shape) - 1) <= max:
            eligible.append(i)
    return eligible


def set_terminals(eligible, n_terminals=100):
        rng = np.random.default_rng()
        terminals = rng.choice(eligible, size=n_terminals, replace=False)
        return list(map(int, terminals))


def create_netlist(n_terminals, length):
        rng = np.random.default_rng()
        netlist = set()
        while len(netlist) != length:
            pair = rng.choice(n_terminals, 2, replace=False)
            if tuple(pair[::-1]) in netlist:
                continue
            netlist.add(tuple(map(int,pair)))
        return list(netlist)


def createdataset(name, shapes, nlayouts, samples_per_layout, netlist_lengths, n_terminals=100, min_available=None):
    dataset = {"netlists": {}, "shapes": {}}

    for length in netlist_lengths:
        dataset["netlists"][f"N{length}"] = \
            [create_netlist(n_terminals, length) for _ in range(samples_per_layout)]

    for shape in shapes:
        eligible = set_eligible_vertices(shape, min_available)
        samples = [set_terminals(eligible, n_terminals) for _ in range(nlayouts)]
        dataset["shapes"][str(shape)] = samples
    
    print("Write File")
    data_path = Path(__file__).parent / f"../../../data/{name}"
    if not data_path.exists():
        os.mkdir(data_path)
    with open(data_path / "test_data.json", "w") as fd:
        json.dump(dataset, fd)


def create_3d_data():
    Z = 7
    dims = list(range(20, 101, 10))
    for d in dims:
        z, y, x = 7, d, d
        p = x * y
        m = p * z
        indices = np.array(range(d * d * Z))

        indices = np.setdiff1d(indices, range(0, x))                        # a - b                 _ - H\
        indices = np.setdiff1d(indices, range(p - x, p))                    # c - d             _ -       \
        indices = np.setdiff1d(indices, range(m - p, m - p + x))            # e - f         G\-         |  \
        indices = np.setdiff1d(indices, range(m - x, m))                    # g - h         | \             \
        indices = np.setdiff1d(indices, range(0, p - x, x))                 # a - c         |  \     _  D  _ \F
        indices = np.setdiff1d(indices, range(x - 1, p, x))                 # b - d         |   \_     _ \    |
        indices = np.setdiff1d(indices, range(m - p, m - x, x))             # e - g         C\   \E_ -        |
        indices = np.setdiff1d(indices, range(m - p + x - 1, m, x))         # f - h           \   |         \ |
        indices = np.setdiff1d(indices, range(0, m - p + 1, p))             # a - e            \  |        _ -B
        indices = np.setdiff1d(indices, range(x - 1, m - p + x, p))         # b - f             \ |    _ -
        indices = np.setdiff1d(indices, range(p - x, m - x + 1, p))         # c - g              \A_ -
        indices = np.setdiff1d(indices, range(p - 1, m, p))                 # d - h
        rng = np.random.default_rng()
        terminals = rng.choice(indices, size=100, replace=False) # Generates the terminals 

        terminal_dict = {"indices": {}, "pathlists": {}, "dims": [d, d]}
        for i, t in enumerate(terminals):
            terminal_dict["indices"][f"g{i}"] = int(t)

        keys = np.array(list(terminal_dict["indices"]))
        for n in range(10, 91):
            l = []
            for i in range(70):
                s = set()
                while n != len(s):
                    s.add(tuple(rng.choice(keys, 2, replace=False)))
                l.append(list(s))
            terminal_dict["pathlists"][f"N{n}"] = l

        print("start write")
        if not Path(f"../data3d/x{d}y{d}/").exists():
            os.mkdir(Path(f"../data3d/x{d}y{d}/"))
        with open(Path(f"../data3d/x{d}y{d}/test_data.json"), "w+") as fd:
            json.dump(terminal_dict, fd)


if __name__ == "__main__":
    shapes = [(x, x, x) for x in range(10, 27, 2)]
    # shapes = [(x, x, 7, 1) for x in range(8, 25, 2)]
    createdataset("orderrouting_sides_long", shapes, 10, 50, range(10, 151), n_terminals=200)
    # create_dataset("testset", shapes, 2, 5, range(10, 13), n_terminals=20)
    # dc = DataCreator(shapes[0])
    # dc.set_eligble_vertices()
    # dc.set_terminals()