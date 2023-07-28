from dataclasses import dataclass
from genericpath import isdir
import os
import sys
import logging
import json
import re

import numpy as np
from time import sleep
from pathlib import Path

DATA_PATH = "data_extended"

def create_3d_data():
    Z = 7
    dims = list(range(20, 101, 10))
    for d in dims:
        z, y, x = 4, d, d
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
            # fd.write(terminal_dict)



def config_logger(config_path = None):
    # Set accurate path here.
    with open("logging/settings.json") as fd:
        logging.config.dictConfig(json.load(fd))

def mesh_reformatter(path):
    d = dict()
    with open(path) as fd:
        text = fd.read()
    vals = ",".join(text.strip().split("\n")).split(",")
    for i, val in enumerate(vals):
        d[val] = i
    return d

def get_pathlists(path):
    l  = []
    for i in range(70):
        with open(path / (path.name + f"_{i}.csv")) as fd:
            vals = ",".join(fd.read().strip().split("\n")).split(",")
        
        # Below can be sped up by not delling but i * 3 + 3...
        l2 = []
        while(vals):
            l2.append(vals[1:3])
            del vals[:3]
        l.append(l2)
    
    return l

def data_formatter():
    # For each x by y mesh create a dictionary containing as key the xy and value mesh, 
    # key with pathlists, values of pathlists is a dictionary.
    # pathlist dict contains keys for all Ns with as name Nxx and value list of all 20 pathlists.
    cwd_absolute = Path(__file__).parent.resolve()
    data_path = cwd_absolute / ".." / ".." / DATA_PATH
    for mesh_size_path in data_path.iterdir():
        if ".DS_Store" in str(mesh_size_path):
            continue
        nodes = mesh_reformatter(mesh_size_path / "C100.csv")
        mesh_dict = {f"indices": nodes, "pathlists": {}}
        mesh_dict["dims"] = list(map(int, list(filter(None, re.split('x|y', mesh_size_path.name)))))

        for pathlist_path in (mesh_size_path / "C100").iterdir():
            mesh_dict["pathlists"][pathlist_path.name] = get_pathlists(pathlist_path)
        json.dump(mesh_dict, open(mesh_size_path / "test_data.json", "w"))

def read_json_data_file(path):
    with open(path, 'r') as fd:
            return json.load(fd)

def data_reader(paths):
    if issubclass(type(paths), (str, Path)):
        return read_json_data_file(paths)
    return [read_json_data_file(path) for path in paths]


def nodes_to_indices(node_dict, pathlists):
    l = []
    for pathlist in pathlists:
        l.append((node_dict[start], node_dict[end]) for start, end in pathlist)
    return l

class WindowsInhibitor:
        '''Prevent OS sleep/hibernate in windows; code from:
        https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
        API documentation:
        https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001

        def __init__(self):
            pass

        def inhibit(self):
            import ctypes
            print("Preventing Windows from going to sleep")
            ctypes.windll.kernel32.SetThreadExecutionState(
                WindowsInhibitor.ES_CONTINUOUS | \
                WindowsInhibitor.ES_SYSTEM_REQUIRED)

        def uninhibit(self):
            import ctypes
            print("Allowing Windows to go to sleep")
            ctypes.windll.kernel32.SetThreadExecutionState(
                WindowsInhibitor.ES_CONTINUOUS)


import json
def reformat_data():
    for i in range(20, 101, 10):
        data_dict = {}
        with open(f"../data/original/x{i}y{i}/test_data.json", "r") as fd:
            a = json.load(fd)
        
        new_indices = []
        for j in range(100):
            new_indices.append(a["indices"][f"g{j}"])
        data_dict["shapes"] = {f"({i}, {i}, 7)": [new_indices]}


        new_dict = {}
        for pathlist in a["pathlists"].keys():
            new_pathlist = []
            for sample in a["pathlists"][pathlist]:
                new_sample = []
                for terminal_pair in sample:
                    new_sample.append([int(x.replace("g", "")) for x in terminal_pair])
                new_pathlist.append(new_sample)
            new_dict[pathlist] = new_pathlist
        data_dict["netlists"] = new_dict

        with open(f"../data/original/x{i}y{i}/test_data_new.json", "w") as fd:
            json.dump(data_dict, fd)

if __name__ == "__main__":
    reformat_data()
    # merge_old_new_results()
    # data_formatter()
    # create_3d_data()