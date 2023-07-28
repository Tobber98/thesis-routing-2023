import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

MESHSIZES = list(range(20, 101, 10))

reitze_results_path = Path("../../Point2PointRoutabilityMod/results")
def main():
    # for m in MESHSIZES:
    #     dims = f"x{m}y{m}"
    #     df = pd.DataFrame()
    #     a = reitze_results_path / dims
    #     for b in a.iterdir():
    #         for c in b.iterdir():
    #             for d in c.iterdir():
                    
    #                 df2 = pd.read_csv(d / "all_data.csv", sep=";", header=None, nrows=1)
    #                 df2.columns = ["id", "routed", 'totlen', '?']
    #                 df2["n"] = int(c.name[1:])
    #                 df2["id"] = int(d.name.split("_")[1])
    #                 df = pd.concat([df, df2])
    #     df_reitze = df[["id", "routed", "n"]]
    #     # print(df_reitze)

    results_paths_max = Path(f"../results/x20y20/np_results.json")
    results_paths_min = Path(f"../results/x20y20/np_results_nf.json")
    dfmax = pd.read_json(results_paths_max, orient="index")
    dfmin = pd.read_json(results_paths_min, orient="index")
    
    for a, b in zip(dfmax["npaths"], dfmin["npaths"]):
        print(np.array(np.array(a) < np.array(b)).sum(), end="\t")
        print(np.array(np.array(a) > np.array(b)).sum())
        
    print(list(dfmax["npaths"] >= dfmin["npaths"]))
        # print(df2)
        # d = {}
        # for n, g in df_reitze.groupby("n"):
        #     for n2, g2 in g.groupby("id"):
        #         row = df2.loc[f"N{n}"]

        #         # print(n , n2, ":", g2["routed"].to_list()[0], row["npaths"][n2])
        #         d[f"{n}-{n2}"] = g2["routed"].to_list()[0] == row["npaths"][n2]
        # # for path in results_paths:
        # # print(d)
        # # print(d.values())
        # print(set(d.values()))
if __name__ == "__main__":
    main()