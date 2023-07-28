import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import itertools

from src.plotter.plotter import MLPGraph
from src.routing.routing import Routing
from src.utils import nodes_to_indices
from copy import deepcopy

MTS = 18
TS = 22
LW = 6
MS = 140


class Plotter2D:
    def __init__(self, zlim, ylim, xlim) -> None:
        self.zlim: int = zlim
        self.ylim: int = ylim
        self.xlim: int = xlim

        self.plot_init()
    
    def plot_init(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim((0, self.xlim))
        ax.set_ylim((0, self.ylim)) 
        ax.set_zlim((0, self.zlim)) 

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        x = np.arange(0, self.xlim + 1, 1)
        y = np.arange(0, self.ylim + 1, 1)
        X, Y = np.meshgrid(x, y)
        Z = X*0.
        
        ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('gist_earth'), alpha=.8)
        for z in range(1, self.zlim):
            ax.plot_surface(X, Y, Z + z, cmap=plt.get_cmap('gist_earth'), alpha=.05)

        plt.axis("off")
        self.ax = ax


    def show(self):
        plt.show()

def plot_order_relevance_example():
    indices = [(8, 26), (19, 22), (12, 11)]
    indices2 = [(19, 22), (8, 26), (12, 11)]
    indices3 =[(12, 11), (19, 22)]
    for i in [indices, indices2, indices3]:
        r = Routing()
        r.set_mesh_dims(1, 6, 6)
        q = r.run3(i)
        print(q)
        grapher = MLPGraph(1, 5, 5)
        grapher.plot_point((0, 1, 2), "blue")
        grapher.plot_point((0, 4, 2), "blue")
        grapher.plot_paths(q)
        grapher.show()
        r.reset_mesh(reset_accessible=True)


def test():
    indices = [(8, 26), (19, 22), (12, 11)]
    indices2 = [(19, 22), (8, 26), (12, 11)]
    indices3 =[(12, 11), (19, 22), (8, 26)]
    
    dc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [dc[0], dc[1], dc[2]]
    colors2 = [dc[1], dc[0], dc[2]]
    colors3 = [dc[2], dc[1], dc[0]]

    # plt.figure(figsize=(10, 10))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.tight_layout()

    for ax, i, j in zip(axes, [indices, indices2, indices3], [colors, colors2, colors3]):
        r = Routing()
        r.set_mesh_dims(1, 6, 6)
        q = r.run3(i)
        length = sum(map(len, q)) - 3 if q[2] else "~"

        ax.set(aspect="equal")
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim([-.2, 5.2])
        ax.set_ylim([-.2, 5.2])
        ax.set_axisbelow(True)
        ax.grid(linewidth = 1.5)
        for i, (a, c) in enumerate(zip(q, j)):
            if a:
                ax.scatter(*list(zip(a[0][1:], a[-1][1:])), c=c, s=MS)
                for g in [a[0][1:], a[-1][1:]]:
                    x, y = g
                    ax.text(x + .1, y + .1,  f"{i + 1}", ha="center", va="center", size=MTS)
            else:
                ax.scatter([1, 4], [2, 2], c=c, s=MS)
                ax.text(1.1, 2.1, "3", ha="center", va="center", size=MTS)
                ax.text(4.1, 2.1, "3", ha="center", va="center", size=MTS)
            b = np.array(list((map(lambda x: x[1:], a))))
            ax.plot(*b.T, lw=LW, c=c)

        ax.text(3.5, .5, f"Total path length: {length}", size=TS, ha="center", va="center", bbox=dict(boxstyle="round",
                   ec=(.5, 0.5, 0.5),
                   fc=(.9, 0.9, 0.9),
                   ))


    plt.savefig("../figures/routing_order_example.pdf")     
    plt.show()


def presentation_plots():
    indices = [[(8, 26), (19, 22), (12, 11)], [(19, 22), (8, 26), (12, 11)], [(12, 11), (19, 22), (8, 26)]]
    terminals = [[(1, 2), (3, 6), (5, 4)], [(3, 6), (1, 2), (5, 4)], [(5, 4), (3, 6), (1, 2)]]
    dc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [[dc[0], dc[1], dc[2]], [dc[1], dc[0], dc[2]], [dc[2], dc[1], dc[0]]]
    plt.rc('axes', axisbelow=True)
    for i1, (i, j, t) in enumerate(zip(indices, colors, terminals)):
        plt.figure(figsize=(10, 10))
        r = Routing()
        r.set_mesh_dims(1, 6, 6)
        coords = list(map(lambda x: (r.index_unravel(x[0])[1:], r.index_unravel(x[1])[1:]), i))
        q = r.run3(i)
        length = sum(map(len, q)) - 3 if q[2] else "~"
        plt.axis("square")
        plt.xticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.yticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.xlim([-.2, 5.2])
        plt.ylim([-.2, 5.2])
        plt.grid(linewidth = 1.5)
        for i2, (a, c, t1) in enumerate(zip(q, j, t)):
            if a:
                plt.scatter(*list(zip(a[0][1:], a[-1][1:])), c=c, s=MS)
                for ind, g in enumerate([a[0][1:], a[-1][1:]]):
                    x, y = g
                    plt.text(x + .1, y + .1,  rf"$T_{t1[ind]}$", ha="center", va="center", size=MTS)
            else:
                plt.scatter([1, 4], [2, 2], c=c, s=MS)
                plt.text(1.1, 2.1, rf"$T_{t1[1]}$", ha="center", va="center", size=MTS)
                plt.text(4.1, 2.1, rf"$T_{t1[0]}$", ha="center", va="center", size=MTS)
        
        plt.savefig(f"../figures/presentation/test_{i1}_terminals.png")
        for i2, (a, c) in enumerate(zip(q, j)):
            b = np.array(list((map(lambda x: x[1:], a))))
            plt.plot(*b.T, lw=LW, c=c)
            plt.savefig(f"../figures/presentation/test_{i1}_{i2}.png")
        plt.text(3.5, .5, f"Total path length: {length}", size=TS, ha="center", va="center", bbox=dict(boxstyle="round",
                ec=(.5, 0.5, 0.5),
                fc=(.9, 0.9, 0.9),
                ))
        plt.savefig(f"../figures/presentation/test_{i1}_final.png", bbox_inches="tight")

# plt.savefig(f"../figures/routing_order_example_presentation_{i}.pdf")     
        plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.axis("square")
    plt.grid(linewidth = 1.5)
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.yticks([0, 1, 2, 3, 4, 5], labels=[5, 4, 3, 2, 1, 0])
    ax.xaxis.tick_top()
    plt.xlim([-.2, 5.2])
    plt.ylim([-.2, 5.2])
    plt.savefig("../figures/presentation/grid_plot.png")
    # plt.show()

    # for a in q:
        # plt.scatter(*list(zip(a[0][1:], a[-1][1:])), c="black", s=MS)
    ncoords = []
    for c in coords:
        ncoords.extend(c)

    # print("Hello", *zip(*ncoords))
    plt.xticks([0, 1, 2, 3, 4, 5], labels=[])
    plt.yticks([0, 1, 2, 3, 4, 5], labels=[])
    plt.scatter(*zip(*ncoords), c="black", s=MS)
    for en, (nc, te) in enumerate(zip(ncoords, [4,5,6,3,2,1])):
        print(en, nc, te)
        plt.text(nc[0] + .1, nc[1] + .1,  rf"$T_{te}$", ha="center", va="center", size=MTS)
    plt.savefig("../figures/presentation/grid_plot_wdots.png")
    plt.show()


def pq_example():
    def plot_points(coords, t_all, color):
        for coord, t, c in zip(coords, t_all, color):
            plt.scatter(*coord, s=MS, c=c)
            x, y = coord
            plt.text(x + .1, y + .1,  rf"$T_{t}$", ha="center", va="center", size=MTS)

    def plot_line(start, end):
        plt.plot(*zip(start, end), lw=LW, c=dc[0], zorder=.91)

    def plot_next_step(coord):
        plt.scatter(*coord, c=dc[3], zorder=.911, s=MS - 50)

    def plot_neighbours(coord, exclude=[]):
        new = []
        if coord[0] < 5 and "r" not in exclude:
            new.append((coord[0] + 1, coord[1]))
        if coord[0] > 0 and "l" not in exclude:
            new.append((coord[0] - 1, coord[1]))
        if coord[1] < 5 and "u" not in exclude:
            new.append((coord[0], coord[1] + 1))
        if coord[1] > 0 and "d" not in exclude:
            new.append((coord[0], coord[1] - 1))
        
        for n in new:
            plt.plot(*zip(coord, n), lw=LW - 2.5, c=dc[3], linestyle="--", zorder=.9)
            plt.scatter(*n, c=dc[3], zorder=.911, s=MS - 50)
        return new

    def reset_plot():
        plt.rc('axes', axisbelow=True)    
        plt.figure(figsize=(10, 10))
        plt.axis("square")
        plt.xticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.yticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.xlim([-.2, 5.2])
        plt.ylim([-.2, 5.2])
        plt.grid(linewidth = 1.5)

    def plot_text(g, h, location):
        plt.text(location[0] + .03, location[1] - .4, f"cost:       {g}\nprediction: {h}\nf-score:    {g + h}",  fontdict={"family": "monospace"})

    def manhattan(g1, g2):
        return abs(g1[0] - g2[0]) + abs(g1[1] - g2[1])

    def save(name):
        plt.savefig(f"../figures/presentation/{name}.png")

    indices = [[(8, 26), (19, 22), (12, 11)], [(19, 22), (8, 26), (12, 11)], [(12, 11), (19, 22), (8, 26)]]
    terminals = [[(2, 1), (6, 3), (4, 5)], [(6, 3), (2, 1), (4, 5)], [(4, 5), (6, 3), (2, 1)]]
    dc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [[dc[0], dc[1], dc[2]], [dc[1], dc[0], dc[2]], [dc[2], dc[1], dc[0]]]
    i, t_all, color = list(itertools.chain(*indices[0])), list(itertools.chain(*terminals[0])), list(itertools.chain(*zip(colors[0], colors[0])))
    router = Routing()
    router.set_mesh_dims(1, 6, 6)
    coords = list(map(lambda x: router.index_unravel(x)[1:], i))
    start, goal = coords[0], coords[1]
    x, y = start

    reset_plot()
    plot_points(coords, t_all, color)
    save("0_initial setup")
    n1 = plot_neighbours(start)
    g = 1 
    for n in n1:
        plot_text(g, manhattan(n, goal), n)
    save("1_neighbours_1")
    # plot_line(start, (x + 1, y))
    # save("2_neighbours_1+chosen")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x + 1, y))
    plot_next_step((x + 1, y))
    save("2_next_node")
    n2 = plot_neighbours((x + 1, y), ["l"])
    g = 2 
    for n in n2:
        plot_text(g, manhattan(n, goal), n)
    save("3_neighbours_2")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x + 2, y))
    plot_next_step((x + 2, y))
    save("4_next_node")
    n3 = plot_neighbours((x + 2, y), ["l", "d"])
    g = 3 
    for n in n3:
        plot_text(g, manhattan(n, goal), n)
    save("5_neighbours_3")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x + 3, y))
    save("6_routed")
    plt.show()


    """
    Bug example plot in 2D
    """

    reset_plot()
    coords2 = [(2,1), (1,2), (3,1), (3,2), (2,2), (2,3)]
    t_all = [1, 2, 3, 4, 5, 6]
    color = [dc[2], dc[2], dc[1], dc[1], dc[0], dc[0]]
    plot_points(coords2, t_all, color)
    plt.plot(*zip((2, 1), (1,1)), lw=LW, c=dc[2], zorder=.91)
    plt.plot(*zip((1, 1), (1,2)), lw=LW, c=dc[2], zorder=.91)
    plt.plot(*zip((3, 1), (3,2)), lw=LW, c=dc[1], zorder=.91)
    plt.plot(*zip((2, 2), (2,3)), lw=LW, c=dc[3], zorder=.91, linestyle="--")
    save("bug_example")
    plt.show()


def pq_example2():
    def plot_points(coords, t_all, color):
        for coord, t, c in zip(coords, t_all, color):
            plt.scatter(*coord, s=MS, c=c)
            x, y = coord
            plt.text(x + .1, y + .1,  rf"$T_{t}$", ha="center", va="center", size=MTS)

    def plot_line(start, end):
        plt.plot(*zip(start, end), lw=LW, c=dc[2], zorder=.91)

    def plot_next_step(coord):
        plt.scatter(*coord, c=dc[3], zorder=.911, s=MS - 50)

    def plot_neighbours(coord, exclude=[]):
        new = []
        if coord[0] < 5 and "r" not in exclude:
            new.append((coord[0] + 1, coord[1]))
        if coord[0] > 0 and "l" not in exclude:
            new.append((coord[0] - 1, coord[1]))
        if coord[1] < 5 and "u" not in exclude:
            new.append((coord[0], coord[1] + 1))
        if coord[1] > 0 and "d" not in exclude:
            new.append((coord[0], coord[1] - 1))
        
        for n in new:
            plt.plot(*zip(coord, n), lw=LW - 2.5, c=dc[3], linestyle="--", zorder=.9)
            plt.scatter(*n, c=dc[3], zorder=.911, s=MS - 50)
        return new

    def reset_plot():
        plt.rc('axes', axisbelow=True)    
        plt.figure(figsize=(10, 10))
        plt.axis("square")
        plt.xticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.yticks([0, 1, 2, 3, 4, 5], labels=[])
        plt.xlim([-.2, 5.2])
        plt.ylim([-.2, 5.2])
        plt.grid(linewidth = 1.5)

    def plot_text(g, h, location):
        plt.text(location[0] + .03, location[1] - .4, f"cost:       {g}\nprediction: {h}\nf-score:    {g + h}",  fontdict={"family": "monospace"})

    def manhattan(g1, g2):
        return abs(g1[0] - g2[0]) + abs(g1[1] - g2[1])

    def save(name):
        plt.savefig(f"../figures/presentation/{name}.png")

    indices = [[(8, 26), (19, 22), (12, 11)], [(19, 22), (8, 26), (12, 11)], [(11, 12), (19, 22), (8, 26)]]
    terminals = [[(2, 1), (6, 3), (4, 5)], [(6, 3), (2, 1), (4, 5)], [(5, 4), (6, 3), (2, 1)]]
    dc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [[dc[0], dc[1], dc[2]], [dc[1], dc[0], dc[2]], [dc[2], dc[1], dc[0]]]
    i, t_all, color = list(itertools.chain(*indices[2])), list(itertools.chain(*terminals[2])), list(itertools.chain(*zip(colors[2], colors[2])))
    router = Routing()
    router.set_mesh_dims(1, 6, 6)
    coords = list(map(lambda x: router.index_unravel(x)[1:], i))
    start, goal = coords[0], coords[1]
    x, y = start

    reset_plot()
    plot_points(coords, t_all, color)
    save("00_initial setup")
    n1 = plot_neighbours(start)
    g = 1 
    for n in n1:
        plot_text(g, manhattan(n, goal), n)
    save("01_neighbours_1")
    # plot_line(start, (x + 1, y))
    # save("2_neighbours_1+chosen")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 1))
    plot_next_step((x, y - 1))
    save("02_next_node")
    n2 = plot_neighbours((x, y - 1), ["u"])
    g = 2 
    for n in n2:
        plot_text(g, manhattan(n, goal), n)
    save("03_neighbours_2")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 2))
    plot_next_step((x, y - 2))
    save("04_next_node")
    n3 = plot_neighbours((x, y - 2), ["u", "d"])
    g = 3 
    for n in n3:
        plot_text(g, manhattan(n, goal), n)
    save("05_neighbours_3")

    
    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 2))
    plot_line((x, y - 2), (x + 1, y - 2))
    plot_next_step((x + 1, y - 2))
    save("06_next_node")
    n3 = plot_neighbours((x + 1, y - 2), ["l"])
    g = 3 
    for n in n3:
        plot_text(g, manhattan(n, goal), n)
    save("07_neighbours_4")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 2))
    plot_line((x, y - 2), (x + 1, y - 2))
    plot_line((x + 1, y - 2), (x + 1, y - 3))
    plot_next_step((x + 1, y - 3))
    save("08_next_node")
    n3 = plot_neighbours((x + 1, y - 3), ["u", "l"])
    g = 3 
    for n in n3:
        plot_text(g, manhattan(n, goal), n)
    save("09_neighbours_4")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 2))
    plot_line((x, y - 2), (x + 1, y - 2))
    plot_line((x + 1, y - 2), (x + 1, y - 3))
    plot_line((x + 1, y - 2), (x + 1, y - 4))
    plot_next_step((x + 1, y - 4))
    save("10_next_node")
    n3 = plot_neighbours((x + 1, y - 4), ["u", "r"])
    g = 3 
    for n in n3:
        plot_text(g, manhattan(n, goal), n)
    save("11_neighbours_4")

    reset_plot()
    plot_points(coords, t_all, color)
    plot_line(start, (x, y - 2))
    plot_line((x, y - 2), (x + 1, y - 2))
    plot_line((x + 1, y - 2), (x + 1, y - 3))
    plot_line((x + 1, y - 2), (x + 1, y - 5))
    save("12_routed")
    plt.show()


    """
    Bug example plot in 2D
    """

    reset_plot()
    coords2 = [(2,1), (1,2), (3,1), (3,2), (2,2), (2,3)]
    t_all = [1, 2, 3, 4, 5, 6]
    color = [dc[2], dc[2], dc[1], dc[1], dc[0], dc[0]]
    plot_points(coords2, t_all, color)
    plt.plot(*zip((2, 1), (1,1)), lw=LW, c=dc[2], zorder=.91)
    plt.plot(*zip((1, 1), (1,2)), lw=LW, c=dc[2], zorder=.91)
    plt.plot(*zip((3, 1), (3,2)), lw=LW, c=dc[1], zorder=.91)
    plt.plot(*zip((2, 2), (2,3)), lw=LW, c=dc[3], zorder=.91, linestyle="--")
    save("bug_example")
    plt.show()


from PIL import Image
from io import BytesIO
def to_gif():
    frames = []
    for filename in range(313): # needs to be extended
        with open(f"brap/a{filename}.png", "rb") as fd:
            data = fd.read()
        imp = BytesIO(data)
        frames.append(Image.open(imp))
    frames[0].save('../figures/presentation/3d_example.gif', format="GIF", save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=2)


def to_animated(format):
    frames = []
    for filename in range(313): # needs to be extended
        with open(f"brap/a{filename}.png", "rb") as fd:
            data = fd.read()
        imp = BytesIO(data)
        frames.append(Image.open(imp))
    frames[0].save(f'../figures/presentation/3d_example.{format}', format=f"{format}", save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=2)


def to_webp():
    from PIL import Image
    frames = []
    for fn in range(313):
        frames.append(Image.open(f"brap/a{fn}.png"))

    frames[0].save('../figures/presentation/3d_example_optimized.webp', format="WEBP", save_all=True, append_images=frames[1:], duration=50, loop=0)


from src.plotter.plotter import PlotlyGraph
import json
from pathlib import Path
def plotly_3d_example():
    def index_unravel(i):
        y, x = 20, 20
        return (i // (y * x), i // x % y, i % x)
    
    with open(Path("../results/original/x20y20/np_results_ext_.json"), "r") as fd:
        results = json.load(fd)

    with open(Path("../data/original/x20y20/test_data.json"), "r") as fd:
        data = json.load(fd)

    data_coords = list(map(index_unravel, data["indices"].values()))

    results_coords = [list(map(index_unravel, x)) for x in results["N20"]["paths"][0]]
    plotter = PlotlyGraph(7, 20, 20)
    plotter.plot_init()
    plotter.animate_paths(results_coords)
    plotter.plot_animation()
    plotter.plot_point(zip(*data_coords), 100)
    plotter.add_terminals(data_coords, 100)
    plotter.add_points()
    
    # plotter.show()
    # plotter.save_initial("2d")
    plotter.save_animation()

def plot_terminal_layout():
    def index_unravel(i):
        y, x = 12, 12
        return (i // (y * x), i // x % y, i % x)
    
    with open(Path("../results/orderrouting_sides_test/results.json"), "r") as fd:
        results = json.load(fd)

    with open(Path("../data/orderrouting_sides_test/test_data.json"), "r") as fd:
        data = json.load(fd)

    index_list, layout = data["netlists"]["N20"][0], data["shapes"]["(12, 12, 12)"][0]
    data_coords = list(map(index_unravel, layout))
    # terminals = [index_unravel(layout[terminal]) for pair in index_list for terminal in pair]
    terminals = [[index_unravel(layout[pair[0]]), index_unravel(layout[pair[1]])] for pair in index_list]
    # data_coords = list(map(index_unravel, data["indices"].values()))

    # results_coords = [list(map(index_unravel, x)) for x in results["N20"]["paths"][0]]
    plotter = PlotlyGraph(12, 12, 12)
    # plotter.plot_init()

    plotter.plot_surfaces()
    plotter.animate_paths(terminals)
    # plotter.plot_animation()
    plotter.plot_point(zip(*data_coords), 100)
    plotter.add_terminals(data_coords, 100)
    plotter.add_points()
    plotter.fig.update_layout(showlegend=False)

    # plotter.show()
    plotter.save_initial("terminal_layout_3d_sides_n12")
    # plotter.save_animation()


def plot_terminal_layout_original():
    def index_unravel(i):
        y, x = 20, 20
        return (i // (y * x), i // x % y, i % x)
    
    with open(Path("../results/original/x20y20/results.json"), "r") as fd:
        results = json.load(fd)

    with open(Path("../data/original/x20y20/test_data_new.json"), "r") as fd:
        data = json.load(fd)

    index_list, layout = data["netlists"]["N20"][0], data["shapes"]["(20, 20, 7)"][0]
    data_coords = list(map(index_unravel, layout))
    # terminals = [index_unravel(layout[terminal]) for pair in index_list for terminal in pair]
    terminals = [[index_unravel(layout[pair[0]]), index_unravel(layout[pair[1]])] for pair in index_list]
    # data_coords = list(map(index_unravel, data["indices"].values()))

    # results_coords = [list(map(index_unravel, x)) for x in results["N20"]["paths"][0]]
    plotter = PlotlyGraph(7, 20, 20)
    # plotter.fig.update_layout(scene_camera_eye=dict(x=7, y=7, z=1))
    # plotter.plot_init()
    plotter.plot_surfaces()
    plotter.animate_paths(terminals)
    # plotter.plot_animation()
    plotter.plot_point(zip(*data_coords), 100)
    plotter.add_terminals(data_coords, 100)
    plotter.add_points()
    plotter.fig.update_layout(showlegend=False)
    # plotter.show()
    plotter.save_initial("terminal_layout_3d_original_n20", original=True)
    # plotter.save_animation()

def crop_png():
    from PIL import Image
    def crop_center(pil_img, crop_width, crop_height):
        img_width, img_height = pil_img.size
        return pil_img.crop(((img_width - crop_width) // 2 - 50,
                            (img_height - crop_height) // 2,
                            (img_width + crop_width) // 2 - 50,
                            (img_height + crop_height) // 2))
    im = Image.open("../figures/terminal_layout/3d_n12.png")
    cropped_im = crop_center(im, 800, 800)
    cropped_im.show()




if __name__ == "__main__":
    # plot_order_relevance_example()
    test()
    presentation_plots()
    # pq_example()
    # pq_example2()
    # plotly_3d_example()
    plot_terminal_layout()
    # plot_terminal_layout_original()
    # crop_png()
    # to_gif()
    # to_apng()
    # to_tiff()
    # to_webp()
    # to_animated("webp")