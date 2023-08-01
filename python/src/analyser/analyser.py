import pandas as pd 
from pandas import DataFrame
import numpy as np
from pathlib import Path
import json

def dict_to_df(ix) -> DataFrame:
    i, x = ix
    partial_df = pd.DataFrame.from_dict(x, orient="index")
    partial_df["layout_index"] = i
    partial_df.reset_index(inplace=True, drop=True)
    partial_df = pd.concat([partial_df.drop(['completed'], axis=1), pd.json_normalize(partial_df["completed"]).add_prefix("completed_")], axis=1)
    partial_df = pd.concat([partial_df.drop(['nroutable'], axis=1), pd.json_normalize(partial_df["nroutable"]).add_prefix("nroutable_")], axis=1)
    partial_df = pd.concat([partial_df.drop(['path_length'], axis=1), pd.json_normalize(partial_df["path_length"]).add_prefix("path_length_")], axis=1)
    return partial_df


def load_new_results_format(path: Path) -> DataFrame:
    with open(path, "r") as fd:
        json_results = json.load(fd)
    partial_dfs = []
    for shape in json_results:
        partial_dfs.extend(map(dict_to_df, enumerate(json_results[shape])))
    return pd.concat(partial_dfs, ignore_index=True)


def analyse_order_dataframe(df: DataFrame) -> DataFrame:
    df["shape_str"]     = df["shape"].apply(str)
    df["volume"]        = df["shape"].apply(np.prod)
    aggregation_functions = {
        'min_distances':        np.hstack,
        "volume":               "first", 
        "shape":                "first"
        }
    nroutable_cols = df.columns[df.columns.str.startswith('nroutable')]
    for c in nroutable_cols:
        c = c.replace("nroutable_", "")
        aggregation_functions[f"nroutable_{c}"] = np.hstack
        aggregation_functions[f"completed_{c}"] = np.hstack
        aggregation_functions[f"path_length_{c}"] = np.hstack

    df = df.groupby(["shape_str", "length"], as_index=False).aggregate(aggregation_functions)

    for c in nroutable_cols:
        c = c.replace("nroutable_", "")
        df[f"nroutable_{c}"]    = df[f"nroutable_{c}"].apply(np.array)
        df[f"routability_{c}"]  = df[f"nroutable_{c}"].floordiv(df["length"])
        df[f"fraction_{c}"]     = df[f"routability_{c}"].apply(np.mean)

        df[f"min_density_{c}"]  = df[f"min_distances"] * df[f"completed_{c}"]#.sum()
        df[f"min_density_{c}"]  = df[f"min_density_{c}"].apply(lambda x: np.sum(x, axis=1)) / df["volume"]
        df[f"density_{c}"]      = (df[f"path_length_{c}"] / df["volume"]).apply(np.mean)
    return df


def analyse_dataframe(df: DataFrame) -> DataFrame:
    df["shape_str"]     = df["shape"].apply(str)
    df["volume"]        = df["shape"].apply(np.prod)
    df["nroutable"]     = df["nroutable"].apply(np.array)

    df["iroutable"]     = df["nroutable"].apply(lambda x: x[:,0])
    df["proutable"]     = df["nroutable"].apply(lambda x: np.max(x, axis=1))

    df["imean"]         = df["iroutable"].apply(np.mean)
    df["pmean"]         = df["proutable"].apply(np.mean)

    df["iroutability"]  = df["iroutable"].floordiv(df["length"])
    df["proutability"]  = df["proutable"].floordiv(df["length"])

    df["ifraction"]      = df["iroutability"].apply(np.mean)
    df["pfraction"]      = df["proutability"].apply(np.mean)

    return df


if __name__ == "__main__":
    df = load_new_results_format(Path("../smallshapes/results_test.json"))
    df = analyse_dataframe(df)