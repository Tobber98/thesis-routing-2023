# Standard lib imports
import os, sys
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

# External lib imports
import numpy as np
import pandas as pd

# Local imports
from src.utils import WindowsInhibitor
from src.plotter.plotter import plot_routability, plot_both, plot_all, plot_new_all_old_curves

# Constants
MESHSIZES = list(range(20, 101, 10))
WORKERS = 1

data_paths = [Path(f"../data/original/x{i}y{i}/test_data.json") for i in MESHSIZES]
results_paths = [Path(f"../results/original/x{i}y{i}/np_results_std.json") for i in MESHSIZES]
results_permuted_paths = [Path(f"../results/original/x{i}y{i}/p_results_std.json") for i in MESHSIZES]

results_paths_zmin = [Path(f"../results/original/x{i}y{i}/np_results_zmin.json") for i in MESHSIZES]
results_permuted_paths_zmin = [Path(f"../results/original/x{i}y{i}/p_results_zmin.json") for i in MESHSIZES]

# results_paths_bug = [Path(f"../results/x{i}y{i}/np_results_bug.json") for i in MESHSIZES]
# results_permuted_paths_bug = [Path(f"../results/x{i}y{i}/p_results_bug.json") for i in MESHSIZES]

results_paths_max_g = [Path(f"../results/x{i}y{i}/np_results_max_g.json") for i in MESHSIZES]
results_permuted_paths_max_g = [Path(f"../results/x{i}y{i}/p_results_max_g.json") for i in MESHSIZES]

# results_paths[0] = Path(f"../results/x20y20/np_results_zmin.json")
# results_permuted_paths[0] = Path(f"../results/x20y20/np_results_zmax.json")

data_3d_paths = [Path(f"../data3d/x{i}y{i}/test_data.json") for i in MESHSIZES]
results_3d_paths = [Path(f"../results3d/x{i}y{i}/np_results2.json") for i in MESHSIZES]
results_3d_permuted_paths = [Path(f"../results3d/x{i}y{i}/p_results2.json") for i in MESHSIZES]

data_extended_paths = [Path(f"../data_extended/x{i}y{i}/test_data.json") for i in MESHSIZES]
results_extended_paths = [Path(f"../results_extended/x{i}y{i}/np_results_bigsample.json") for i in MESHSIZES]
results_extended_permuted_paths = [Path(f"../results/x{i}y{i}/p_results.json") for i in MESHSIZES]

def run(path):
    extension = ".exe" if os.name == "nt" else ""
    rust_router = Path(f"../rust/routing/target/release/routing{extension}")
    print(f"Starting process for file: {path}")
    with open("out.txt", "w") as out:
        subprocess.run([rust_router, path, '-s', '0', '-p', '200', '-c', f"{os.cpu_count() // WORKERS - 2}", "-e", "std"], stdout=sys.stdout)
    print(f"Finished process for file: {path}")    
    return path

def analyse_results(paths, name):
    df = pd.DataFrame()
    for path in paths:
        df2 = pd.read_json(path, orient="index")
        df = pd.concat([df, df2])
    df["npaths"]        = df["npaths"].apply(np.array)
    df["max_paths"]     = df["npaths"].apply(lambda x: np.max(x, axis=0) if x.ndim > 1 else x)
    df["mean_paths"]    = df["max_paths"].apply(np.mean)
    df["length"]        = df.index.map(lambda x: int(x[1:]))
    df["routability"]   = df["max_paths"].floordiv(df["length"])
    df["fraction"]      = df["routability"].apply(np.mean)
    df["dims"]          = df["dims"].apply(tuple)
    df.to_pickle(f"temp_{name}.pkl")
    grouped_df          = df[["dims", "fraction", "length"]].groupby("dims")
    return grouped_df

def main(args):
    osSleep = None
    if args.run:
        if os.name == "nt":
            osSleep = WindowsInhibitor()
            osSleep.inhibit()
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            executor.map(run, data_paths[::-1])
    
    if args.run3d:
        if os.name == "nt":
            osSleep = WindowsInhibitor()
            osSleep.inhibit()
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            executor.map(run, data_3d_paths[::-1])
    
    if args.analyse:
        std_group = analyse_results(results_paths, "np")
        per_group = analyse_results(results_permuted_paths, "p")
        std_group_min = analyse_results(results_paths_zmin, "np")
        per_group_min = analyse_results(results_permuted_paths_zmin, "p")
        # plot_routability("2d", std_group, per_group)
        # plot_both(std_group, per_group)
        plot_new_all_old_curves(std_group, per_group)

        # std_group_min = analyse_results(results_paths_max_g, "np")
        # per_group_min = analyse_results(results_permuted_paths_max_g, "p")
        plot_all("zmin", std_group, per_group, std_group_min, per_group_min)

    if args.analyse3d:
        std_group = analyse_results(results_paths, "np")
        per_group = analyse_results(results_permuted_paths, "p")
        std_3d_group = analyse_results(results_3d_paths, "np")
        per_3d_group = analyse_results(results_3d_permuted_paths, "p")
        # plot_routability("3d", std_3d_group, per_3d_group)
        plot_all("3d_all", std_group, per_group, std_3d_group, per_3d_group)
    if osSleep:
        osSleep.uninhibit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run", action="store_true")
    parser.add_argument("-a", "--analyse", action="store_true")
    parser.add_argument("-s", "--run3d", action="store_true")
    parser.add_argument("-b", "--analyse3d", action="store_true")
    # main(vars(parser.parse_args()))
    main(parser.parse_args())
