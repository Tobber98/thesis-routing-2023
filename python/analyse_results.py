# from argparse import ArgumentParser
import sys
import tomllib
from pathlib import Path

from src.analyser.analyser import *
from src.plotter.plotter import new_routability, new_rpoint, new_slope

def main(fp):
    with open(fp, "rb") as fd:
        config = tomllib.load(fd)
    
    df = load_new_results_format(config["4d"]["results_path"])
    df = analyse_dataframe(df)
    print(df)
    groups = df[["volume", "shape", "ifraction", "pfraction", "length"]].groupby("volume")
    # for group in groups:
    #     print(group)
    #     exit()
    # plot_routability("new_data_test", groups, groups)
    p = new_routability(groups)
    new_rpoint(p)
    new_slope(p)

if __name__ == "__main__":
    path = Path(__file__).parent / "config.toml"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    main(path)