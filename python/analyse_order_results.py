# from argparse import ArgumentParser
import sys
import tomllib
from pathlib import Path

from src.analyser.analyser import *
from src.plotter.plotter import new_routability, order_rpoint, new_slope, plot_order, plot_order_modular, performance_compare_table


from src.analyser.analyser import load_new_results_format
import pandas as pd


global_mse = 0

def statistical_test(a, b):
    from scipy.stats import ttest_ind, probplot, wilcoxon
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Test normality first (unknown name + distribution plot of datapoint)
    # if normal -> ttest
    # else test name forgotten
    # print(a.iloc[0:81])
    print(wilcoxon(a.iloc[0:81], b.iloc[0:81]))
    # print(ttest_ind(a, b))
    # sns.displot(a.iloc[0:81], kind="kde")
    # plt.show()
    probplot(a.iloc[0:81], dist="norm", plot=plt)
    # scipy.stats.distributions.
    plt.show()

def relative_performance(baseline_col, df):
    baseline = np.concatenate(df[f"routability_{baseline_col}"].values)

    colnames = df.columns[df.columns.str.startswith('routability')]
    ud = {sum(baseline < np.concatenate(df[colname].values)): colname for colname in colnames}
    print(f"{'name':>14} | {'better':^16} | {'same':^16} | {'worse':^16} | {'total':^10}\n{'#'*82}")
    for key, item in sorted(ud.items()):
        test = np.concatenate(df[item].values)
        name = item.replace("routability_", "")
        print(f"{name:>14} | \
{sum(baseline < test) / len(baseline) * 100:6.2f}% ({sum(baseline < test):>6}) | \
{sum(baseline == test) / len(baseline) * 100:6.2f}% ({sum(baseline == test):>6}) | \
{sum(baseline > test) / len(baseline) * 100:6.2f}% ({sum(baseline > test):>6}) | \
{len(baseline):^10}")
        
        # print(f"{name} & \\makecell{{{sum(baseline < test) / len(baseline) * 100:6.2f}\\% \\\\({sum(baseline < test):>6})}} & \makecell{{{sum(baseline == test) / len(baseline) * 100:6.2f}\\% \\\\({sum(baseline == test):>6})}} &  \makecell{{{sum(baseline > test) / len(baseline) * 100:6.2f}\\% \\\\({sum(baseline > test):>6})}} \\\\")
        
def relative_performance_adjusted(baseline_col, df):
    colnames = df.columns[df.columns.str.startswith('fraction')]
    df = df[df[colnames].eq(df[colnames].iloc[:, 0], axis=0).all(1).apply(lambda x: not x)]
    
    baseline = np.concatenate(df[f"routability_{baseline_col}"].values)

    colnames = df.columns[df.columns.str.startswith('routability')]
    ud = {sum(baseline < np.concatenate(df[colname].values)): colname for colname in colnames}
    print(f"{'name':>14} | {'better':^16} | {'same':^16} | {'worse':^16} | {'total':^10}\n{'#'*82}")
    for key, item in sorted(ud.items()):
        test = np.concatenate(df[item].values)
        name = item.replace("routability_", "")
        print(f"{name: >14} | \
{sum(baseline < test) / len(baseline) * 100:6.2f}% ({sum(baseline < test):>6}) | \
{sum(baseline == test) / len(baseline) * 100:6.2f}% ({sum(baseline == test):>6}) | \
{sum(baseline > test) / len(baseline) * 100:6.2f}% ({sum(baseline > test):>6}) | \
{len(baseline):^10}")


def analyse_dataframe(df, name):
    print(f"\n{'#'*56}\n{name:^56}\n{'#'*56}")
    # print(df.loc[:, df.columns.str.startswith('fraction')].mean() * 100)
    # print(df.loc[:, df.columns.str.startswith('density')].replace(0, np.NaN).mean(skipna=True) * 100)
    
    # relative_performance("random", df)
    # relative_performance_adjusted("random", df)
    # relative_performance("longest", df)
    # relative_performance_adjusted("longest", df)

    performance_compare_table(df, name=name, exclude=["routability_adaptive_max_density_shortest", "routability_adaptive_max_density_longest", "routability_adaptive_max_density_min_volume"])

    # print(df[["fraction_random", "fraction_shortest", "fraction_longest", "fraction_min_z", "fraction_min_y", "fraction_min_x"]].mean() * 100)
    # groups = df[["volume", "shape", "fraction_random", "fraction_shortest", "fraction_longest", "length"]].groupby("volume")
    # params = plot_order(groups, name=name)
    # order_rpoint(params, name)

    groups = df.groupby("volume")
    params = plot_order_modular(groups, name=name, cols=["fraction_random", "fraction_shortest", "fraction_longest", "fraction_adaptive_max_density_random", "fraction_max_density"])
    order_rpoint(params, f"{name}_mod")


def merge_old_new_results():
    all = []
    for i in range(20, 101, 10):
        with open(f"../results/original/x{i}y{i}/results.json", "r") as fd:
            json_result = json.load(fd)

        all.append(load_new_results_format(f"../results/original/x{i}y{i}/results.json"))
    df = pd.concat(all, ignore_index=True)
    df = analyse_order_dataframe(df)

    analyse_dataframe(df, "original")


def analyse_results_file(path):
    name = "_".join(path.split("/")[2].split("_")[1:])

    df = load_new_results_format(path)
    df = analyse_order_dataframe(df)

    analyse_dataframe(df, name)


def main(fp):
    with open(fp, "rb") as fd:
        config = tomllib.load(fd)
    paths = config["order_results"]["results_path"]
    if type(paths) != str:
        for path in paths:
            analyse_results_file(path)
    else:
        analyse_results_file(paths)    

if __name__ == "__main__":
    path = Path(__file__).parent / "config.toml"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    main(path)
    merge_old_new_results()