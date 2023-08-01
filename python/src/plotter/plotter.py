# imports
import logging
from pathlib import Path
import time
from dataclasses import dataclass
from itertools import cycle

# Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

# import kaleido
import plotly
import plotly.graph_objects as go
import sympy

import scipy
from scipy.optimize import curve_fit
from math import sqrt, ceil

# Local imports

# Desmos.com

# Constants
FIGURES_PATH = Path("../figures")

try:
    from tkinter import Tk
    a = Tk(useTk=False)
    xres, yres = a.winfo_screenwidth() - 100, a.winfo_screenheight() - 200
    a.destroy()
except:
    xres, yres = 1800, 1000


@dataclass
class LineCoords:
    x: list
    y: list
    z: list
    length: int
    color_index: int


class MLPGraph:
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

        ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('gist_earth'), alpha=.3)
        for z in range(1, self.zlim):
            ax.plot_surface(
                X, Y, Z + z, cmap=plt.get_cmap('gist_earth'), alpha=.05)

        plt.axis("off")
        for i in range(self.xlim):
            plt.plot((i, i), (0, self.ylim), (0, 0), color="black", lw=.5)
        for i in range(self.ylim):
            plt.plot((0, self.xlim), (i, i), (0, 0), color="black", lw=.5)

        self.ax = ax

    def plot_point(self, coordinates, color):
        self.ax.scatter(*coordinates[::-1], color=color)

    def plot_line(self, coordinates_start, coordinates_end, color):
        print(*zip(coordinates_start[::-1], coordinates_end[::-1]))
        plt.plot(
            *zip(coordinates_start[::-1], coordinates_end[::-1]), color=color, alpha=1)

    def plot_path(self, path, color):
        current = path.pop()
        self.plot_point(current, color)
        while path:
            next = path.pop()
            self.plot_line(current, next, color)
            current = next
        self.plot_point(current, color)

    def plot_paths(self, paths):
        colors = ["red", "green", "blue"]
        for path, color in zip(paths, colors):
            self.plot_path(path, color)

    def show(self):
        plt.show()


class PlotlyGraph:
    def __init__(self, zlim, ylim, xlim) -> None:
        self.zlim = zlim
        self.ylim = ylim
        self.xlim = xlim

        self.colors = plotly.colors.qualitative.Plotly
        self.ncolors = len(self.colors)

        self.plot_init()

        # Used for the animation
        self.lines = []
        self.points_to_plot = []

    def plot_init(self):
        self.fig = go.Figure(data=[go.Scatter3d()])
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(
                    nticks=self.xlim,
                    range=[0, self.xlim - 1],
                    showticklabels=False,
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(64, 64, 64, 70)"
                ),
                yaxis=dict(
                    nticks=self.ylim,
                    range=[0, self.ylim - 1],
                    showticklabels=False,
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(64, 64, 64, 70)"
                ),
                zaxis=dict(
                    nticks=self.zlim,
                    range=[0, self.zlim - 1],
                    showticklabels=False,
                    showgrid=True,
                    gridcolor="rgba(64, 64, 64, 70)",
                ),
                aspectratio=dict(x=self.xlim / 4,
                                 y=self.ylim / 4, z=self.zlim / 4),
                xaxis_showspikes=False,
                yaxis_showspikes=False,
                zaxis_showspikes=False,
                zaxis_title=dict(text="Layer", font_size=24),
                yaxis_title=dict(font_size=24),
                xaxis_title=dict(font_size=24)
            ),
            scene_camera = dict(
                up=dict(x=1, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=5.5, y=5.5, z=2)
            ),       
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            # scene_camera_eye=dict(x=1.25, y=1.25, z=1.25),
            # width=yres,
            # height=yres,
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            # updatemenus=[dict(
            #     type="buttons",
            #     buttons=[dict(label="Play",
            #                   method="animate",
            #                   args=[None])])]
        )


    def plot_surfaces(self):
        x = np.arange(0, self.xlim + 1, 1)
        y = np.arange(0, self.ylim + 1, 1)
        X, Y = np.meshgrid(x, y)
        Z = X*0.

        for z in range(1, self.zlim):
            # self.fig.add_trace(go.Surface(x=X, y=Y, z=Z+z, showlegend=False, opacity=0.05, colorscale="greys"))

            for i in range(self.xlim):
                self.fig.add_trace(go.Scatter3d(x=(i, i), y=(0, self.ylim), z=(z, z), mode="lines", line=dict(color="black", width=2), showlegend=False, opacity=0.1))
            for i in range(self.ylim):
                self.fig.add_trace(go.Scatter3d(x=(0, self.xlim), y=(i, i), z=(z, z), mode="lines", line=dict(color="black", width=2), showlegend=False, opacity=0.1))


    def plot_point(self, coordinates, index):
        z, y, x = coordinates

        self.fig.add_trace(
            go.Scatter3d(
                x=[x] if type(x) != tuple else x,
                y=[y] if type(y) != tuple else y,
                z=[z] if type(z) != tuple else z,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.colors[index % self.ncolors]
                ),
                showlegend=False,
                hoverinfo='none',
                legendgroup=index,
                name=f"trace {index}"
            )
        )

    def plot_line(self, path, index):
        z, y, x = list(zip(*path))
        self.fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=self.colors[index % self.ncolors],
                    width=4
                ),
                legendgroup=index,
                name=f"trace {index}"
            )
        )
        # if len(self.fig.frames) == 0:
        #     self.fig.frames = [go.Frame(data=
        #     go.Scatter3d(
        #         x=x,
        #         y=y,
        #         z=z,
        #         mode="lines",
        #         line=dict(
        #             color=self.colors[index % self.ncolors],
        #             width=4
        #         ),
        #         legendgroup=index,
        #         name=f"trace {index}"
        #     )
        # )]
        # else:
        #     self.fig.frames = [*self.fig.frames, go.Frame(
        #         data=[
        #             *self.fig.frames[-1].data,
        #             go.Scatter3d(
        #                 x=x,
        #                 y=y,
        #                 z=z,
        #                 mode="lines",
        #                 line=dict(
        #                     color=self.colors[index % self.ncolors],
        #                     width=4
        #                 ),
        #                 legendgroup=index,
        #                 name=f"trace {index}"
        #             )
        #         ]
        #     )
        # ]

    def plot_path(self, path, index):
        self.plot_point(path[0], index)
        self.plot_point(path[-1], index)
        self.plot_line(path, index)

    def plot_paths(self, paths):
        for i, path in enumerate(paths):
            self.plot_path(path, i)

    def show(self):
        self.fig.show()

    def reset(self):
        self.fig.data = []

    """

    Animation code for plotly (work in progress)
    
    """

    def plot_animation(self):
        from copy import deepcopy

        cycler = []
        x_eye = -1.25
        y_eye = 2
        z_eye = 0.5

        def rotate_z(x, y, z, theta):
            w = x+1j*y
            return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
        for t in np.arange(0, 6.26, 0.02):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
            cycler.append({"x": xe, "y": ye, "z": ze})
        pool = cycle(cycler)

        base_class = {
            "type": "scatter3d",
            "x": [],
            "y": [],
            "z": [],
            "mode": "lines",
        }
        base = [base_class for _ in range(len(self.lines))]
        base2 = deepcopy(base)
        f = []

        for j, line in enumerate(self.lines):
            for i in range(line.length):
                x = line.x[0: i + 1]
                y = line.y[0: i + 1]
                z = line.z[0: i + 1]
                base[j] = {
                    "type": "scatter3d",
                    "x": x,
                    "y": y,
                    "z": z,
                    "mode": "lines",
                    "line": {
                        "color": self.colors[line.color_index % self.ncolors],
                        "width": 4
                    },
                    "legendgroup": line.color_index,
                    "name": f"trace {line.color_index}"
                }
                base2[j] = {
                    "type": "scatter3d",
                    "mode": "lines",
                    "line": {
                        "color": self.colors[line.color_index % self.ncolors],
                        "width": 4
                    },
                    "legendgroup": line.color_index,
                    "name": f"trace {line.color_index}"
                }
                f.append(
                    dict(data=deepcopy(base),
                         layout={"scene_camera_eye": next(pool)}
                         )
                )
        self.fig = go.Figure(
            data=base2,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(
                        nticks=self.xlim,
                        range=[0, self.xlim - 1],
                        backgroundcolor="rgba(0,0,0,0)",
                        showticklabels=False,
                        gridcolor="rgba(64, 64, 64, 70)"
                    ),
                    yaxis=dict(
                        nticks=self.ylim,
                        range=[0, self.ylim - 1],
                        backgroundcolor="rgba(0,0,0,0)",
                        showticklabels=False,
                        gridcolor="rgba(64, 64, 64, 70)"
                    ),
                    zaxis=dict(
                        nticks=self.zlim,
                        range=[0, self.zlim - 1],
                        showgrid=True,
                        showticklabels=False,
                        gridcolor="rgba(64, 64, 64, 70)"
                    ),
                    aspectratio=dict(x=self.xlim / 12,
                                     y=self.ylim / 12, z=self.zlim / 12),
                    xaxis_showspikes=False,
                    yaxis_showspikes=False,
                    zaxis_showspikes=False,
                    # scene_camera_eye = dict(x = -1.25, y = 2, z = 0.5)
                ),
                scene_camera_eye=dict(x=5, y=5, z=1),
                width=xres,
                height=yres,
                autosize=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                # updatemenus=[dict(
                #     type="buttons",
                #     buttons=[dict(label="Play",
                #                   method="animate",
                #                   args=[None, {"frame": {"duration": 10}, "fromcurrent": True, "transition": {"duration": 0}}])])],
                geo=dict(projection_type="equirectangular",
                         visible=True, resolution=50)

            ),
            frames=f
        )

    def save_initial(self, name, original=False):
        from PIL import Image
        from io import BytesIO
        import plotly.io as pio

        # d = self.fig.frames[0].to_plotly_json()
        # self.fig.update(d)
        # self.plot_point(self.terminals[:3], self.terminals[3])
        # self.add_points()
        w = 1000
        if original:
            w = 1400
        self.fig.write_image(f"../figures/terminal_layout/{name}.png", format="png", height=1080, width=w, scale=1)
        self.fig.write_image(f"../figures/terminal_layout/{name}.svg", format="svg", height=1080, width=w, scale=1)

    def save_animation(self):
        from PIL import Image
        from io import BytesIO
        import plotly.io as pio

        # terminals = go.Scatter3d(
        #         x=[x] if type(x) != tuple else x,
        #         y=[y] if type(y) != tuple else y,
        #         z=[z] if type(z) != tuple else z,
        #         mode='markers',
        #         marker=dict(
        #             size=4,
        #             color=self.colors[index % self.ncolors]
        #         ),
        #         showlegend=False,
        #         hoverinfo='none',
        #         legendgroup=index,
        #         name=f"trace {index}"
        #     )

        frames = []
        for i, frame in enumerate(self.fig.frames):
            print(f"frame: {i}")
            d = frame.to_plotly_json()
            self.fig.update(d)
            self.plot_point(self.terminals[:3], self.terminals[3])
            self.add_points()
            frames.append(Image.open(BytesIO(pio.to_image(
                self.fig, format="png", scale=1, width=2560, height=1440))))
        
            # a = Image.open(BytesIO(self.fig.to_image(format="svg", height=1440*2, width=2560*2)))
            self.fig.write_image(f"../figures/animation/{i}.png", format="png", height=1080, width=1920)
            # a.save(f"brap/{i}.png")

        frames[0].save('../figures/presentation/3d_example2.gif', format="GIF",
                       save_all=True, append_images=frames[1:], duration=50, loop=0, disposal=2)
        print("Save is successful")
        # images.append(PIL.Image.open(io.BytesIO(self.fig.to_image(format="png"))))
        # images[0].save(
        # "test.gif",
        # save_all=True,
        # append_images=images[1:],
        # optimize=True,
        # duration=500,
        # loop=0,
    # )

    def add_line_to_animation(self, path, index):
        z, y, x = list(zip(*path))
        lc = LineCoords(x, y, z, len(z), index)
        self.lines.append(lc)

    def animate_path(self, path, index):
        self.points_to_plot.append((path[0], index))
        self.points_to_plot.append((path[-1], index))
        # self.plot_point(path[0], index)
        # self.plot_point(path[-1], index)
        self.add_line_to_animation(path, index)

    def add_terminals(self, data_coords, index):
        self.terminals = (*zip(*data_coords), index)

    def add_points(self):
        for p, i in self.points_to_plot:
            self.plot_point(p, i)

    def animate_paths(self, paths):
        for i, path in enumerate(paths):
            self.animate_path(path, i)


def sigmoid(x, x0, k, b=None):
    return 1 / (1 + np.exp(-k*(x-x0)))


def inverse_sigmoid(y, x0, k, b=None):
    return np.log((1 - y) / (y)) / -k + x0


def log_func(x, a, b, c):
    return a * np.log(x * b - c)
    # return a * np.log(np.where(x*b < 0, -1*(x*b), x*b)) + c


def print_log_func(a, b, c):
    sympy.init_printing(use_unicode=True)
    x = sympy.symbols('x')
    d = sympy.Mul(sympy.Float(a, 3), sympy.ln(sympy.Add(sympy.Mul(sympy.Float(b, 3), x), -sympy.Float(c, 3), evaluate=False)), evaluate=False)
    return sympy.pretty(d, full_prec=False)


def get_mse(x, y, func, popt):
    return np.mean((y-func(x, *popt))**2)

def get_rmse(x, y, func, popt):
    residuals = y - func(x, *popt)
    return (scipy.sum(residuals ** 2) / (residuals.size - 2)) ** 0.5

def get_mae(x, y, func, popt):
    return np.mean(np.abs(y - func(x , *popt)))

def get_curve(x, ydata, func, slope=False):
    if func.__name__ == "sigmoid":
        p0 = [np.median(x), -1]
    elif not slope:
        p0 = [13, 0.005, -0.2]
    else: 
        # p0 = [-0.8, 0.0005, 0.0001]
        p0 = [-0.1, 0.0001, -0.03]
    popt, _ = curve_fit(func, x, ydata, p0, maxfev=10000)
    return popt


def save_fig(name, s):
    path = FIGURES_PATH / name / s
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        path / "{}_{:4}-{:02}-{:02}_{:02}.svg".format(s, *time.gmtime()), dpi=200, bbox_inches="tight", pad_inches=.05)
    plt.savefig(
        path / "{}_{:4}-{:02}-{:02}_{:02}.png".format(s, *time.gmtime()), dpi=200, bbox_inches="tight", pad_inches=.05)
    plt.savefig(
        path / "{}_{}.svg".format(name.split("/")[-1].split("_")[0], s), dpi=200, bbox_inches="tight", pad_inches=.05)


def set_plt_params(s, m, l):
    plt.rcParams.update({
        "font.size": s,
        "axes.titlesize": s,
        "axes.labelsize": l,
        "xtick.labelsize": s,
        "ytick.labelsize": s,
        "legend.fontsize": m,
        "figure.titlesize": l
    })


def plot_points_and_curve(ax, group, color: str, curve_func=None, color2=None, label=None):
    if type(group) == tuple or type(group) == list:
        a, b = group
    else:
        a, b = group["length"], group["fraction"]
    ax.plot(a, b, linestyle="None", marker="o", ms=5,
            color=color, mew=0, markerfacecoloralt=color2, label = label)
    if curve_func:
        ax.plot(a, b, linestyle="None", marker="o", ms=5, color=color, mew=0)
        curve_params = get_curve(a, b, curve_func)
        ax.plot(a, curve_func(a.to_list(), *curve_params),
                linestyle="--", color="black", linewidth=1)
        return np.array(curve_params)
    else:
        ax.plot(a, b, linestyle="None", marker="o", ms=5, color=color,
                mew=0, markerfacecoloralt=color2, fillstyle="left")
        return None


def plot_routability_point(rp_data, params, c, l, b, scatter = True, ls="-"):
    d1 = list(map(lambda x: x * x, list(params)))
    dims = list(range(min(d1), max(d1) + 1))
    curve_params = get_curve(d1, rp_data, log_func)
    if scatter:
        plt.scatter(d1, rp_data, color=c, label=l)
        plt.plot(dims, log_func(np.array(dims), *curve_params),
                linestyle=ls, color=c, linewidth=1, label=l+"2")
    else:
        plt.plot(dims, log_func(np.array(dims), *curve_params),
                linestyle=ls, color=c, linewidth=1, label=l)
    plt.text(6400, 35 + b, rf"{print_log_func(*curve_params)}", fontsize="x-large")
    # ax.text(4900, 33.8, print_log_func(*curve))
    # print_log_func(*curve_params)
    return np.array(curve_params)


def plot_slope(rp_data, params, c, l, scatter=True, ls="-", b = 0):
    d1 = list(map(lambda x: x * x, list(params)))
    dims = list(range(min(d1), max(d1) + 1))
    rp_data = np.array(rp_data) * -1 # Should we keep this

    curve_params = get_curve(d1, rp_data, log_func, slope=True)
    if scatter:
        plt.scatter(d1, rp_data, color=c, label=l)
        plt.plot(dims, log_func(np.array(dims), *curve_params),
             linestyle=ls, color=c, linewidth=1, label=l+"2")
    else:
        plt.plot(dims, log_func(np.array(dims), *curve_params),
             linestyle=ls, color=c, linewidth=1, label=l)
    plt.text(6400, b, rf"{print_log_func(*curve_params)}", fontsize="x-large")
    return np.array(curve_params)


def plot_routabilty_corr(x1, y1, x2, y2, name="default"):
    SMALL = 9
    MEDIUM = 11
    LARGE = 16
    LOC_ROUTABILITY_POINT = 2
    BBOX_TO_ANCHOR_ROUTABILITY_POINT = (0.05, 0.5, .9, .45)

    set_plt_params(SMALL, MEDIUM, LARGE)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.08, left=0.03)
    axes = axes.flatten()

    params = {}
    params2 = {}
    for i, (sx, sy, px, py) in enumerate(zip(x1, y1, x2, y2)):
        ax = axes[i]

        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(np.arange(0, 1.1, .1))
        if not i // 6:
            ax.set_xticks([])

        same_x, same_y = sx[sy == py], sy[sy == py]
        std_curve_params = plot_points_and_curve(
            ax, [sx, sy], colors[0], sigmoid)
        perm_curve_params = plot_points_and_curve(
            ax, [px, py], colors[1], sigmoid)
        plot_points_and_curve(ax, [same_x, same_y],
                              colors[0], color2=colors[1])

        d = i * 10 + 20
        params2[d] = (std_curve_params[1], perm_curve_params[1])
        params[d] = (inverse_sigmoid(.5, *std_curve_params),
                     inverse_sigmoid(.5, *perm_curve_params))
        # ax.text(85, .95, f"{d}x{d}", fontsize="x-large")
        ax.text(.995, .99, f"{d}x{d}", fontsize="x-large", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

    fig.legend(["Initial", "Permuted"], bbox_to_anchor=[.5, .025, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="large")
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize="x-large")
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize="x-large")
    save_fig(name, "routability")
    plt.show()


def plot_routability(name: str, std_data: DataFrameGroupBy, per_data: DataFrameGroupBy):
    SMALL = 8
    MEDIUM = 10
    LARGE = 16
    LOC_ROUTABILITY_POINT = 2
    BBOX_TO_ANCHOR_ROUTABILITY_POINT = (0.05, 0.5, .9, .45)

    set_plt_params(SMALL, MEDIUM, LARGE)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.08, left=0.03)
    axes = axes.flatten()

    df = pd.DataFrame(columns=["dims", "perm", "sigmoid_params", "rpoint", "mse"])
    df["perm"] = df["perm"].astype(bool)
    for i, ((d, s), (_, p)) in enumerate(zip(std_data, per_data)):
        ax = axes[i]

        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(np.arange(0, 1.1, .1))
        if not i // 6:
            ax.set_xticks([])

        same = (s[(s["fraction"] == p["fraction"])])
        std_curve_params = plot_points_and_curve(ax, s, colors[0], sigmoid, label="Initial")
        perm_curve_params = plot_points_and_curve(ax, p, colors[1], sigmoid, label="Permuted")
        plot_points_and_curve(ax, same, colors[0], color2=colors[1])

        print(std_curve_params, perm_curve_params)

        mse_std = get_rmse(s["length"], s["fraction"], sigmoid, std_curve_params)
        mse_perm = get_rmse(p["length"], p["fraction"], sigmoid, perm_curve_params)
        print(mse_std, mse_perm)

        df = pd.concat([df,
                        pd.DataFrame([
                            [d[0], False, std_curve_params,
                                inverse_sigmoid(.5, *std_curve_params), mse_std],
                            [d[0], True, perm_curve_params, inverse_sigmoid(.5, *perm_curve_params), mse_perm]],
                            columns=["dims", "perm", "sigmoid_params", "rpoint", "mse"])
                        ],
                       ignore_index=True)

        # ax.text(85, .95, f"{d[0]}x{d[0]}", fontsize="x-large")
        ax.text(.995, .99, f"{d[0]}x{d[0]}", fontsize="x-large", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels , bbox_to_anchor=[.5, .025, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="large")
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize="x-large")
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize="x-large")
    save_fig(name, "routability")
    plt.show()

    # Second plot
    df_std = df[df["perm"] == False]
    df_perm = df[df["perm"] == True]

    plt.figure(figsize=(12, 8))
    plt.ylim = (0, 80)
    plt.yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    rpoint_std_params = plot_routability_point(df_std["rpoint"],
                           df_std["dims"], colors[0], "Initial", 0)
    rpoint_perm_params = plot_routability_point(df_perm["rpoint"],
                           df_perm["dims"], colors[1], "Permuted", 30)
    
    d1 = np.array(list(map(lambda x: x * x, list(df_std["dims"]))))
    mse_std_rpoint = get_rmse(d1, df_std["rpoint"], log_func, rpoint_std_params)
    mse_perm_rpoint = get_rmse(d1, df_perm["rpoint"], log_func, rpoint_perm_params)
    print(mse_std_rpoint, mse_perm_rpoint)


    plt.xlabel("Mesh Area")
    plt.ylabel("Routability Point")
    plt.legend(bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT, fancybox=False, shadow=False, ncol=1, frameon=False)
    save_fig(name, "rpoint")
    plt.show()

    # Third plot
    plt.figure(figsize=(12, 8))
    slope_std_params = plot_slope(np.array(df_std["sigmoid_params"].to_list())[:, 1], 
               df_std["dims"], colors[0], "Initial", b=.15)
    slope_perm_params = plot_slope(np.array(df_perm["sigmoid_params"].to_list())[:, 1], 
               df_perm["dims"], colors[1], "Permuted", b=.23)
    
    mse_std_slope = get_rmse(d1, np.array(df_std["sigmoid_params"].to_list())[:, 1], log_func, slope_std_params)
    mse_perm_slope = get_rmse(d1, np.array(df_perm["sigmoid_params"].to_list())[:, 1], log_func, slope_perm_params)
    print(mse_std_slope, mse_perm_slope)

    plt.xlabel("Mesh Area")
    plt.ylabel("Slope")
    plt.legend(bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT - 1, fancybox=False, shadow=False, ncol=1, frameon=False)
    save_fig(name, "slope")
    plt.show()


def plot_both(std_data: DataFrameGroupBy, per_data: DataFrameGroupBy):
    SMALL = 8
    MEDIUM = 10
    LARGE = 16
    LOC_ROUTABILITY_POINT = 2
    BBOX_TO_ANCHOR_ROUTABILITY_POINT = (0.05, 0.5, .9, .45)

    plt.rcParams.update({
        "font.size": SMALL,
        "axes.titlesize": SMALL,
        "axes.labelsize": LARGE,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": MEDIUM,
        "figure.titlesize": LARGE
    })

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.08, left=0.03)
    axes = axes.flatten()
    
    df = pd.DataFrame(columns=["dims", "perm", "sigmoid_params", "rpoint"])
    df["perm"] = df["perm"].astype(bool)

    df_reitze = pd.read_csv(
        "../../../Point2PointRoutability/compare_routability_best_of_200.csv", index_col="netlist length")

    for i, ((d, s), (_, p)) in enumerate(zip(std_data, per_data)):
        ax = axes[i]

        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])

        same = (s[(s["fraction"] == p["fraction"])])
        ax.plot(s["length"], s["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[0], mew=0)
        ax.plot(p["length"], p["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[1], mew=0)
        ax.plot(same["length"], same["fraction"], linestyle="None", marker="o", ms=5,
                fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)

        std_curve_params = get_curve(s["length"], s["fraction"], sigmoid)
        perm_curve_params = get_curve(p["length"], p["fraction"], sigmoid)
        df = pd.concat([df,
                        pd.DataFrame([
                            [d[0], False, std_curve_params,
                                inverse_sigmoid(.5, *std_curve_params)],
                            [d[0], True, perm_curve_params, inverse_sigmoid(.5, *perm_curve_params)]],
                            columns=["dims", "perm", "sigmoid_params", "rpoint"])
                        ],
                       ignore_index=True)

        ax.plot(s["length"], sigmoid(list(range(10, 91)), *
                std_curve_params), linestyle="--", color="black", linewidth=1)
        ax.plot(p["length"], sigmoid(list(range(10, 91)), *
                perm_curve_params), linestyle="--", color="black", linewidth=1)
    save_fig("2d_both", "routability_own")
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = axes.flatten()

    df2 = pd.DataFrame(columns=["dims", "perm", "sigmoid_params", "rpoint"])
    df2["perm"] = df2["perm"].astype(bool)

    for i, ax in enumerate(axes):
        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])
        dims = i * 10 + 20
        a, b = df_reitze[f"routability by arb {dims}x{dims}"], df_reitze[f"routability best of {dims}x{dims}"]
        same2 = df_reitze[a == b]

        l = np.array(range(len(b))) + 10
        ax.plot(l, a, linestyle="None", marker="o",
                ms=5, color=colors[0], mew=0)
        ax.plot(l, b, linestyle="None", marker="o",
                ms=5, color=colors[1], mew=0)
        ax.plot(same2.index.to_list(), same2[f"routability by arb {dims}x{dims}"], linestyle="None",
                marker="o", ms=5, fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)

        std_curve_params2 = get_curve(list(range(10, 91)), a, sigmoid)
        perm_curve_params2 = get_curve(list(range(10, 91)), b, sigmoid)
        df2 = pd.concat([df2,
                        pd.DataFrame([
                            [dims, False, std_curve_params2,
                                inverse_sigmoid(.5, *std_curve_params2)],
                            [dims, True, perm_curve_params2, inverse_sigmoid(.5, *perm_curve_params2)]],
                            columns=["dims", "perm", "sigmoid_params", "rpoint"])
                        ],
                       ignore_index=True)

        ax.plot(l, sigmoid(list(range(10, 91)), *std_curve_params2),
                linestyle="--", color="black", linewidth=1)
        ax.plot(l, sigmoid(list(range(10, 91)), *perm_curve_params2),
                linestyle="--", color="black", linewidth=1)
    save_fig("2d_both", "routability_reitze")
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.08, left=0.03)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])
        dims = i * 10 + 20

        a = df[(df["dims"] == dims) & ~df["perm"]]["sigmoid_params"].to_list()[0]
        b = df[(df["dims"] == dims) & df["perm"]]["sigmoid_params"].to_list()[0]

        c = df2[(df2["dims"] == dims) & ~df2["perm"]]["sigmoid_params"].to_list()[0]
        d = df2[(df2["dims"] == dims) & df2["perm"]]["sigmoid_params"].to_list()[0]

        # ax.text(85, .95, f"{dims}x{dims}", fontsize="x-large")
        ax.text(.995, .99, f"{dims}x{dims}", fontsize="x-large", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        n1, = ax.plot(l, sigmoid(list(range(10, 91)), *a),
                      linestyle="--", color=colors[0], linewidth=1)
        n2, = ax.plot(l, sigmoid(list(range(10, 91)), *b),
                      linestyle="-", color=colors[0], linewidth=1)
        o1, = ax.plot(l, sigmoid(list(range(10, 91)), *c),
                      linestyle="--", color=colors[1], linewidth=1)
        o2, = ax.plot(l, sigmoid(list(range(10, 91)), *d),
                      linestyle="-", color=colors[1], linewidth=1)
    fig.legend([o1, o2, n1, n2], ["Original Initial", "Original Permuted", "New Initial", "New Permuted"], bbox_to_anchor=[.5, .025, 0, 0],
               loc=8, fancybox=False, shadow=False, ncol=4, frameon=False, fontsize="large")
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize="x-large")
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize="x-large")
    save_fig("2d_both", "routability_both")
    plt.show()



    """
    ######
    Plot Routability Point comparison here!
    ######
    """

    plt.figure(figsize=(12, 8))
    plt.ylim = (0, 80)
    # plt.yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    df_std_new = df[df["perm"] == False]
    df_perm_new = df[df["perm"] == True]
    rpoint_std_params_new = plot_routability_point(df_std_new["rpoint"],
                           df_std_new["dims"], colors[0], "New Initial", 10, scatter=False)
    rpoint_perm_params_new = plot_routability_point(df_perm_new["rpoint"],
                           df_perm_new["dims"], colors[0], "New Permuted", 42, scatter=False, ls="--")
    print(df_perm_new["rpoint"].to_numpy() / df_std_new["rpoint"].to_numpy())
    df_std_original = df2[df2["perm"] == False]
    df_perm_original = df2[df2["perm"] == True]
    rpoint_std_params_original = plot_routability_point(df_std_original["rpoint"],
                           df_std_original["dims"], colors[1], "Original Initial", 0, scatter=False)
    rpoint_perm_params_original = plot_routability_point(df_perm_original["rpoint"],
                           df_perm_original["dims"], colors[1], "Original Permuted", 30, scatter=False, ls="--")

    print(df_perm_new["rpoint"].to_numpy(), df_std_new["rpoint"].to_numpy())
    print(df_perm_original["rpoint"].to_numpy(), df_std_original["rpoint"].to_numpy())

    plt.xlabel("Mesh Area")
    plt.ylabel("Routability Point")
    plt.legend(bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT, fancybox=False, shadow=False, ncol=1, frameon=False)
    save_fig("2d_both", "rpoint")
    plt.show()


    d1 = np.array(list(map(lambda x: x * x, list(df_std_new["dims"]))))
    mse_std_rpoint_new = get_rmse(d1, df_std_new["rpoint"], log_func, rpoint_std_params_new)
    mse_perm_rpoint_new = get_rmse(d1, df_perm_new["rpoint"], log_func, rpoint_perm_params_new)
    mse_std_rpoint_original = get_rmse(d1, df_std_original["rpoint"], log_func, rpoint_std_params_original)
    mse_perm_rpoint_original = get_rmse(d1, df_perm_original["rpoint"], log_func, rpoint_perm_params_original)
    
    print("Rpoint parameters: ")
    print(f"\tstd new: {rpoint_std_params_new}\n\tperm new: {rpoint_perm_params_new}\n\tstd original: {rpoint_std_params_original}\n\tperm original: {rpoint_perm_params_original}")
    print(f"\tstd new: {print_log_func(*rpoint_std_params_new)}\n\tperm new: {print_log_func(*rpoint_perm_params_new)}\n\tstd original: {print_log_func(*rpoint_std_params_original)}\n\tperm original: {print_log_func(*rpoint_perm_params_original)}")

    print("Rpoint MSE: ")
    print(f"\tstd new: {mse_std_rpoint_new}\n\tperm new: {mse_perm_rpoint_new}\n\tstd original: {mse_std_rpoint_original}\n\tperm original: {mse_perm_rpoint_original}")

    # print(df_perm["rpoint"].to_numpy() / df_std["rpoint"].to_numpy())




    """
    ######
    Plot Routability Slope comparison here!
    ######
    """


    plt.figure(figsize=(12, 8))
    slope_std_params_new = plot_slope(np.array(df_std_new["sigmoid_params"].to_list())[:, 1], 
               df_std_new["dims"], colors[0], "Initial New", scatter=False, b=0.155)
    slope_perm_params_new = plot_slope(np.array(df_perm_new["sigmoid_params"].to_list())[:, 1], 
               df_perm_new["dims"], colors[0], "Permuted New", scatter=False, ls="--", b=.24)
    
    slope_std_params_original = plot_slope(np.array(df_std_original["sigmoid_params"].to_list())[:, 1], 
               df_std_original["dims"], colors[1], "Initial Original", scatter=False, b=.12)
    slope_perm_params_original = plot_slope(np.array(df_perm_original["sigmoid_params"].to_list())[:, 1], 
               df_perm_original["dims"], colors[1], "Permuted Original", scatter=False, ls="--", b=.175)
    
    mse_std_slope_new = get_rmse(d1, -1 * np.array(df_std_new["sigmoid_params"].to_list())[:, 1], log_func, slope_std_params_new)
    mse_perm_slope_new = get_rmse(d1, -1 * np.array(df_perm_new["sigmoid_params"].to_list())[:, 1], log_func, slope_perm_params_new)
    mse_std_slope_original = get_rmse(d1, -1 * np.array(df_std_original["sigmoid_params"].to_list())[:, 1], log_func, slope_std_params_original)
    mse_perm_slope_original = get_rmse(d1, -1 * np.array(df_perm_original["sigmoid_params"].to_list())[:, 1], log_func, slope_perm_params_original)


    print("slope parameters: ")
    print(f"\tstd new: {slope_std_params_new}\n\tperm new: {slope_perm_params_new}\n\tstd original: {slope_std_params_original}\n\tperm original: {slope_perm_params_original}")
    print(f"\tstd new: {print_log_func(*slope_std_params_new)}\n\tperm new: {print_log_func(*slope_perm_params_new)}\n\tstd original: {print_log_func(*slope_std_params_original)}\n\tperm original: {print_log_func(*slope_perm_params_original)}")

    print("slope MSE: ")
    print(f"\tstd new: {mse_std_slope_new}\n\tperm new: {mse_perm_slope_new}\n\tstd original: {mse_std_slope_original}\n\tperm original: {mse_perm_slope_original}")

    # print(f"std new: {mse_std_slope_new}\nperm new: {mse_perm_slope_new}\nstd original: {mse_std_slope_original}\n perm original: {mse_perm_slope_original}")

    plt.xlabel("Mesh Area")
    plt.ylabel("Slope")
    plt.legend(bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT - 1, fancybox=False, shadow=False, ncol=1, frameon=False)
    save_fig("2d_both", "slope")
    plt.show()

def plot_new_all_old_curves(std_data: DataFrameGroupBy, per_data: DataFrameGroupBy):
    SMALL = 8
    MEDIUM = 10
    LARGE = 16
    LOC_ROUTABILITY_POINT = 2
    BBOX_TO_ANCHOR_ROUTABILITY_POINT = (0.05, 0.5, .9, .45)

    plt.rcParams.update({
        "font.size": SMALL,
        "axes.titlesize": SMALL,
        "axes.labelsize": LARGE,
        "xtick.labelsize": MEDIUM,
        "ytick.labelsize": MEDIUM,
        "legend.fontsize": MEDIUM,
        "figure.titlesize": LARGE
    })

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # axes = axes.flatten()
    
    df = pd.DataFrame(columns=["dims", "perm", "sigmoid_params", "rpoint"])
    df["perm"] = df["perm"].astype(bool)

    df_reitze = pd.read_csv(
        "../../Point2PointRoutability/compare_routability_best_of_200.csv", index_col="netlist length")

    for i, ((d, s), (_, p)) in enumerate(zip(std_data, per_data)):
        # ax = axes[i]

        # if not i % 3 == 0:
        #     ax.set_yticks([])
        # else:
        #     ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        # if not i // 6:
        #     ax.set_xticks([])

        # same = (s[(s["fraction"] == p["fraction"])])
        # ax.plot(s["length"], s["fraction"], linestyle="None",
        #         marker="o", ms=5, color=colors[0], mew=0)
        # ax.plot(p["length"], p["fraction"], linestyle="None",
        #         marker="o", ms=5, color=colors[1], mew=0)
        # ax.plot(same["length"], same["fraction"], linestyle="None", marker="o", ms=5,
        #         fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)

        std_curve_params = get_curve(s["length"], s["fraction"], sigmoid)
        perm_curve_params = get_curve(p["length"], p["fraction"], sigmoid)
        df = pd.concat([df,
                        pd.DataFrame([
                            [d[0], False, std_curve_params,
                                inverse_sigmoid(.5, *std_curve_params)],
                            [d[0], True, perm_curve_params, inverse_sigmoid(.5, *perm_curve_params)]],
                            columns=["dims", "perm", "sigmoid_params", "rpoint"])
                        ],
                       ignore_index=True)

        # ax.plot(s["length"], sigmoid(list(range(10, 91)), *
        #         std_curve_params), linestyle="--", color="black", linewidth=1)
        # ax.plot(p["length"], sigmoid(list(range(10, 91)), *
        #         perm_curve_params), linestyle="--", color="black", linewidth=1)
    # save_fig("2d_both", "routability_own")
    # plt.show()

    # fig, axes = plt.subplots(3, 3, figsize=(24, 12))
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # axes = axes.flatten()

    df2 = pd.DataFrame(columns=["dims", "perm", "sigmoid_params", "rpoint"])
    df2["perm"] = df2["perm"].astype(bool)

    for i, ax in enumerate(range(9)):
        # if not i % 3 == 0:
        #     ax.set_yticks([])
        # else:
        #     ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        # if not i // 6:
        #     ax.set_xticks([])
        dims = i * 10 + 20
        a, b = df_reitze[f"routability by arb {dims}x{dims}"], df_reitze[f"routability best of {dims}x{dims}"]
        # same2 = df_reitze[a == b]

        # l = np.array(range(len(b))) + 10
        # ax.plot(l, a, linestyle="None", marker="o",
        #         ms=5, color=colors[0], mew=0)
        # ax.plot(l, b, linestyle="None", marker="o",
        #         ms=5, color=colors[1], mew=0)
        # ax.plot(same2.index.to_list(), same2[f"routability by arb {dims}x{dims}"], linestyle="None",
        #         marker="o", ms=5, fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)

        std_curve_params2 = get_curve(list(range(10, 91)), a, sigmoid)
        perm_curve_params2 = get_curve(list(range(10, 91)), b, sigmoid)
        df2 = pd.concat([df2,
                        pd.DataFrame([
                            [dims, False, std_curve_params2,
                                inverse_sigmoid(.5, *std_curve_params2)],
                            [dims, True, perm_curve_params2, inverse_sigmoid(.5, *perm_curve_params2)]],
                            columns=["dims", "perm", "sigmoid_params", "rpoint"])
                        ],
                       ignore_index=True)

        # ax.plot(l, sigmoid(list(range(10, 91)), *std_curve_params2),
        #         linestyle="--", color="black", linewidth=1)
        # ax.plot(l, sigmoid(list(range(10, 91)), *perm_curve_params2),
        #         linestyle="--", color="black", linewidth=1)
    # save_fig("2d_both", "routability_reitze")
    # plt.show()

    l = np.array(range(len(b))) + 10
    fig, axes = plt.subplots(3, 3, figsize=(22, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.11, left=0.038)
    axes = axes.flatten()
    for i, (ax, (_, s), (_, p)) in enumerate(zip(axes, std_data, per_data)):
        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])
        dims = i * 10 + 20

        a = df[(df["dims"] == dims) & ~df["perm"]]["sigmoid_params"].to_list()[0]
        b = df[(df["dims"] == dims) & df["perm"]]["sigmoid_params"].to_list()[0]

        c = df2[(df2["dims"] == dims) & ~df2["perm"]]["sigmoid_params"].to_list()[0]
        d = df2[(df2["dims"] == dims) & df2["perm"]]["sigmoid_params"].to_list()[0]

        # ax.text(85, .95, f"{dims}x{dims}", fontsize="xx-large")
        ax.text(.995, .99, f"{dims}x{dims}", fontsize=18, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        
        
        same = (s[(s["fraction"] == p["fraction"])])
        an, = ax.plot(s["length"], s["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[0], mew=0)
        bn, = ax.plot(p["length"], p["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[1], mew=0)
        ax.plot(same["length"], same["fraction"], linestyle="None", marker="o", ms=5,
                fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)
        
        
        n1, = ax.plot(l, sigmoid(list(range(10, 91)), *a),
                      linestyle="-", color=colors[0], linewidth=1)
        n2, = ax.plot(l, sigmoid(list(range(10, 91)), *b),
                      linestyle="-", color=colors[1], linewidth=1)
        o1, = ax.plot(l, sigmoid(list(range(10, 91)), *c),
                      linestyle="--", color=colors[0], linewidth=1)
        o2, = ax.plot(l, sigmoid(list(range(10, 91)), *d),
                      linestyle="--", color=colors[1], linewidth=1)
    fig.legend([an, bn, n1, n2, o1, o2], ["", "", "New Initial", "New Permuted", "Original Initial", "Original Permuted"], bbox_to_anchor=[.5, .025, 0, 0],
               loc=8, fancybox=False, shadow=False, ncol=3, frameon=False, fontsize="xx-large", borderpad=0.3, handletextpad=.3, columnspacing=0.5)
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize=18)
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize=18)
    save_fig("2_both+scatter", "routability_both")
    plt.show()


    """
    ######
    Plot Routability Point comparison here!
    ######
    """
    plt.figure(figsize=(12, 8))
    plt.ylim = (0, 80)
    df_std_new = df[df["perm"] == False]
    df_perm_new = df[df["perm"] == True]
    rpoint_std_params_new = plot_routability_point(df_std_new["rpoint"],
                           df_std_new["dims"], colors[0], "New Initial", 10, scatter=True)
    rpoint_perm_params_new = plot_routability_point(df_perm_new["rpoint"],
                           df_perm_new["dims"], colors[1], "New Permuted", 42, scatter=True)
    print(df_perm_new["rpoint"].to_numpy() / df_std_new["rpoint"].to_numpy())
    df_std_original = df2[df2["perm"] == False]
    df_perm_original = df2[df2["perm"] == True]
    rpoint_std_params_original = plot_routability_point(df_std_original["rpoint"],
                           df_std_original["dims"], colors[0], "Original Initial", 0, scatter=False, ls="--")
    rpoint_perm_params_original = plot_routability_point(df_perm_original["rpoint"],
                           df_perm_original["dims"], colors[1], "Original Permuted", 30, scatter=False, ls="--")

    print(df_perm_new["rpoint"].to_numpy(), df_std_new["rpoint"].to_numpy())
    print(df_perm_original["rpoint"].to_numpy(), df_std_original["rpoint"].to_numpy())

    plt.xlabel("Mesh Area", fontsize="x-large")
    plt.ylabel("Routability Point", fontsize="x-large")
    hs, ls = plt.gca().get_legend_handles_labels()
    handles = [hs[0], hs[2], hs[1], hs[3], hs[4], hs[5]]
    plt.legend(handles=handles, labels = ["", "", "New Initial", "New Permuted", "Original Initial", "Original Permuted"], bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT, fancybox=False, shadow=False, ncol=3, frameon=False, fontsize="x-large", borderpad=0.3, handletextpad=.3, columnspacing=0.5)
    save_fig("2_both+scatter", "rpoint")
    plt.show()

    d1 = np.array(list(map(lambda x: x * x, list(df_std_new["dims"]))))
    mse_std_rpoint_new = get_rmse(d1, df_std_new["rpoint"], log_func, rpoint_std_params_new)
    mse_perm_rpoint_new = get_rmse(d1, df_perm_new["rpoint"], log_func, rpoint_perm_params_new)
    mse_std_rpoint_original = get_rmse(d1, df_std_original["rpoint"], log_func, rpoint_std_params_original)
    mse_perm_rpoint_original = get_rmse(d1, df_perm_original["rpoint"], log_func, rpoint_perm_params_original)
    
    print("Rpoint parameters: ")
    print(f"\tstd new: {rpoint_std_params_new}\n\tperm new: {rpoint_perm_params_new}\n\tstd original: {rpoint_std_params_original}\n\tperm original: {rpoint_perm_params_original}")
    print(f"\tstd new: {print_log_func(*rpoint_std_params_new)}\n\tperm new: {print_log_func(*rpoint_perm_params_new)}\n\tstd original: {print_log_func(*rpoint_std_params_original)}\n\tperm original: {print_log_func(*rpoint_perm_params_original)}")

    print("Rpoint MSE: ")
    print(f"\tstd new: {mse_std_rpoint_new}\n\tperm new: {mse_perm_rpoint_new}\n\tstd original: {mse_std_rpoint_original}\n\tperm original: {mse_perm_rpoint_original}")


    """
    ######
    Plot Routability Slope comparison here!
    ######
    """
    plt.figure(figsize=(12, 8))
    slope_std_params_new = plot_slope(np.array(df_std_new["sigmoid_params"].to_list())[:, 1], 
               df_std_new["dims"], colors[0], "Initial New", scatter=True, b=0.155)
    slope_perm_params_new = plot_slope(np.array(df_perm_new["sigmoid_params"].to_list())[:, 1], 
               df_perm_new["dims"], colors[1], "Permuted New", scatter=True, b=.24)
    
    slope_std_params_original = plot_slope(np.array(df_std_original["sigmoid_params"].to_list())[:, 1], 
               df_std_original["dims"], colors[0], "Initial Original", scatter=False, ls="--", b=.11)
    slope_perm_params_original = plot_slope(np.array(df_perm_original["sigmoid_params"].to_list())[:, 1], 
               df_perm_original["dims"], colors[1], "Permuted Original", scatter=False, ls="--", b=.175)
    
    mse_std_slope_new = get_rmse(d1, -1 * np.array(df_std_new["sigmoid_params"].to_list())[:, 1], log_func, slope_std_params_new)
    mse_perm_slope_new = get_rmse(d1, -1 * np.array(df_perm_new["sigmoid_params"].to_list())[:, 1], log_func, slope_perm_params_new)
    mse_std_slope_original = get_rmse(d1, -1 * np.array(df_std_original["sigmoid_params"].to_list())[:, 1], log_func, slope_std_params_original)
    mse_perm_slope_original = get_rmse(d1, -1 * np.array(df_perm_original["sigmoid_params"].to_list())[:, 1], log_func, slope_perm_params_original)


    print("slope parameters: ")
    print(f"\tstd new: {slope_std_params_new}\n\tperm new: {slope_perm_params_new}\n\tstd original: {slope_std_params_original}\n\tperm original: {slope_perm_params_original}")
    print(f"\tstd new: {print_log_func(*slope_std_params_new)}\n\tperm new: {print_log_func(*slope_perm_params_new)}\n\tstd original: {print_log_func(*slope_std_params_original)}\n\tperm original: {print_log_func(*slope_perm_params_original)}")

    print("slope MSE: ")
    print(f"\tstd new: {mse_std_slope_new}\n\tperm new: {mse_perm_slope_new}\n\tstd original: {mse_std_slope_original}\n\tperm original: {mse_perm_slope_original}")

    plt.xlabel("Mesh Area")
    plt.ylabel("Slope")
    hs, ls = plt.gca().get_legend_handles_labels()
    handles = [hs[0], hs[2], hs[1], hs[3], hs[4], hs[5]]
    plt.legend(handles=handles, labels = ["", "", "New Initial", "New Permuted", "Original Initial", "Original Permuted"], bbox_to_anchor=BBOX_TO_ANCHOR_ROUTABILITY_POINT,
               loc=LOC_ROUTABILITY_POINT -1, fancybox=False, shadow=False, ncol=3, frameon=False, fontsize="x-large", borderpad=0.3, handletextpad=.3, columnspacing=0.5)
    save_fig("2_both+scatter", "slope")
    plt.show()


def plot_all(name: str, std_data: DataFrameGroupBy, per_data: DataFrameGroupBy, std_data3d: DataFrameGroupBy, per_data3d: DataFrameGroupBy):
    SMALL = 9
    MEDIUM = 11
    LARGE = 16

    plt.rcParams.update({
        "font.size": SMALL,
        "axes.titlesize": MEDIUM,
        "axes.labelsize": LARGE,
        "xtick.labelsize": MEDIUM,
        "ytick.labelsize": MEDIUM,
        "legend.fontsize": MEDIUM,
        "figure.titlesize": LARGE
    })

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(3, 3, figsize=(22, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.11, left=0.038)
    axes = axes.flatten()

    params = {}
    params2 = {}
    params3 = {}

    for i, ((d, s), (_, p)) in enumerate(zip(std_data, per_data)):
        ax = axes[i]

        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])

        same = (s[(s["fraction"] == p["fraction"])])
        ax.plot(s["length"], s["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[0], mew=0)
        ax.plot(p["length"], p["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[1], mew=0)
        ax.plot(same["length"], same["fraction"], linestyle="None", marker="o", ms=5,
                fillstyle='left', color=colors[0], markerfacecoloralt=colors[1], mew=0)

        std_curve_params = get_curve(s["length"], s["fraction"], sigmoid)
        perm_curve_params = get_curve(p["length"], p["fraction"], sigmoid)
        params2[d[0]] = (std_curve_params, perm_curve_params)

        params[d[0]] = (inverse_sigmoid(.5, *std_curve_params),
                        inverse_sigmoid(.5, *perm_curve_params))

        ax.plot(s["length"], sigmoid(list(range(10, 91)), *
                std_curve_params), linestyle="--", color="black", linewidth=1)
        ax.plot(p["length"], sigmoid(list(range(10, 91)), *
                perm_curve_params), linestyle="--", color="black", linewidth=1)

        dims = i * 10 + 20
        # ax.text(85, .95, f"{dims}x{dims}", fontsize="x-large")
        ax.text(.995, .99, f"{dims}x{dims}", fontsize=18, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')


    for i, ((d, s), (_, p)) in enumerate(zip(std_data3d, per_data3d)):
        ax = axes[i]

        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])

        same = (s[(s["fraction"] == p["fraction"])])
        ax.plot(s["length"], s["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[2], mew=0)
        ax.plot(p["length"], p["fraction"], linestyle="None",
                marker="o", ms=5, color=colors[3], mew=0)
        ax.plot(same["length"], same["fraction"], linestyle="None", marker="o", ms=5,
                fillstyle='left', color=colors[2], markerfacecoloralt=colors[3], mew=0)

        std_curve_params = get_curve(s["length"], s["fraction"], sigmoid)
        perm_curve_params = get_curve(p["length"], p["fraction"], sigmoid)
        params3[d[0]] = (std_curve_params, perm_curve_params)

        params[d[0]] = (inverse_sigmoid(.5, *std_curve_params),
                        inverse_sigmoid(.5, *perm_curve_params))

        ax.plot(s["length"], sigmoid(list(range(10, 91)), *
                std_curve_params), linestyle="--", color="black", linewidth=1)
        ax.plot(p["length"], sigmoid(list(range(10, 91)), *
                perm_curve_params), linestyle="--", color="black", linewidth=1)

        dims = i * 10 + 20
        # ax.text(85, .95, f"{dims}x{dims}", fontsize="x-large")
        ax.text(.995, .99, f"{dims}x{dims}", fontsize=18, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')


    fig.legend(["Initial", "Permuted"], bbox_to_anchor=[.5, .025, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="xx-large")
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize=18)
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize=18)
    save_fig(name, "routability")
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(22, 12))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.11, left=0.038)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if not i % 3 == 0:
            ax.set_yticks([])
        else:
            ax.set_yticks(list(map(lambda x:  x/10, range(11))))
        if not i // 6:
            ax.set_xticks([])
        dims = i * 10 + 20
        a = params2[dims][0]
        b = params2[dims][1]

        c = params3[dims][0]
        d = params3[dims][1]

        l = list(range(10, 91))
        # ax.text(85, .95, f"{dims}x{dims}", fontsize="x-large")
        ax.text(.995, .99, f"{dims}x{dims}", fontsize=18, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')

        n1, = ax.plot(l, sigmoid(list(range(10, 91)), *a),
                      linestyle="--", color=colors[0], linewidth=1)
        n2, = ax.plot(l, sigmoid(list(range(10, 91)), *b),
                      linestyle="-", color=colors[0], linewidth=1)
        o1, = ax.plot(l, sigmoid(list(range(10, 91)), *c),
                      linestyle="--", color=colors[1], linewidth=1)
        o2, = ax.plot(l, sigmoid(list(range(10, 91)), *d),
                      linestyle="-", color=colors[1], linewidth=1)
    fig.legend([o1, o2, n1, n2], ["Min-z Initial", "Min-z Permuted", "Max-z Initial", "Max-z Permuted"], bbox_to_anchor=[.5, .025, 0, 0],
            loc=8, fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="xx-large")
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize=18)
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize=18)
    save_fig(name, "routability_both")
    plt.show()


def nrows_and_ncols(n: int):
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    return rows, cols


def set_figure(nplots: int, figsize=(18, 10)):
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    nrows, ncols = nrows_and_ncols(nplots)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.1, left=0.05)
    axes: list(plt.Axes) = axes.flatten()
    return fig, axes


def plot_curve(ax, x, y, curve_func, color, ls="-", label=None):
    params = get_curve(x, y, curve_func)
    x = np.array(range(int(min(x)), int(max(x)) + 1))
    ax.plot(x, curve_func(x, *params), linestyle=ls, color=color, linewidth=1, label=label)
    return np.array(params)


def plot_routability_new(ax, id, group, curve_func=None):
    length, init, perm = group[["length","ifraction", "pfraction"]].T.values
    same = init == perm

    c1, c2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2]
    kwargs = dict(linestyle="None", marker="o", mew=0, ms=5)

    ax.plot(length, init, color=c1, label="Initial", **kwargs)
    ax.plot(length, perm, color=c2, label="Permuted", **kwargs)
    ax.plot(length[same], init[same], color=c1, markerfacecoloralt=c2, fillstyle="left", **kwargs)

    if curve_func:
        init_params = plot_curve(ax, length, init, curve_func, c1)
        perm_params = plot_curve(ax, length, perm, curve_func, c2)
    
    return pd.DataFrame([[init_params, perm_params]], columns=["init", "perm"], index=[id])


def plot_against_volume(init, perm, volume):
    kwargs = dict(linestyle="None", marker="o", mew=0, ms=5)
    fig, axes = set_figure(1)
    ax: plt.Axes = axes[0]

    c1, c2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2]
    ax.plot(volume, init, c1, **kwargs, label="Initial")
    ax.plot(volume, perm, c2, **kwargs, label="Permuted")
    plot_curve(ax, volume, init, log_func, c1)
    plot_curve(ax, volume, perm, log_func, c2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels , bbox_to_anchor=[.5, .025, 0, 0], loc=8,
            fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="large")
    plt.show()


def new_routability(groups: DataFrameGroupBy):
    fig, axes = set_figure(groups.ngroups)

    df_params = pd.concat([plot_routability(ax, id, group, sigmoid) 
                           for ax, (id, group) in zip(axes, groups)])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=[.5, .025, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=2, frameon=False, fontsize="large")
    plt.show()
    return df_params


def new_rpoint(df_params: pd.DataFrame):
    init, perm = df_params[["init", "perm"]].applymap(lambda x: inverse_sigmoid(.5, *x)).T.values
    volume = df_params.index.to_numpy()
    plot_against_volume(init, perm, volume)


def new_slope(df_params: pd.DataFrame):
    init, perm = df_params[["init", "perm"]].applymap(lambda x: x[1]).T.values
    volume = df_params.index.to_numpy()
    plot_against_volume(init, perm, volume)


def plot_order_routability(ax, id, group, curve_func):
    length, random, shortest, longest = group[["length", "fraction_random", "fraction_shortest", "fraction_longest"]].T.values

    c1, c2, c3 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:3]
    kwargs = dict(linestyle="None", marker="o", mew=0, ms=5)

    ax.plot(length, random, color=c1, label="Random", **kwargs)
    ax.plot(length, shortest, color=c2, label="Shortest", **kwargs)
    ax.plot(length, longest, color=c3, label="Longest", **kwargs)

    if curve_func:
        random_params = plot_curve(ax, length, random, curve_func, c1)
        shortest_params = plot_curve(ax, length, shortest, curve_func, c2)
        longest_params = plot_curve(ax, length, longest, curve_func, c3)
    print(f"Routability MSE - {id}")
    print(f"  MSE {'Random:':<30} ({id:>6})\n    {get_rmse(length, random, curve_func, random_params)}")
    print(f"  MSE {'Shortest:':<30} ({id:>6})\n    {get_rmse(length, shortest, curve_func, shortest_params)}")
    print(f"  MSE {'Longest:':<30} ({id:>6})\n    {get_rmse(length, longest, curve_func, longest_params)}")
    
    shape = group["shape"].iloc[0]
    ax.text(.995, .99, f"{shape}", fontsize="x-large", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    return pd.DataFrame([[random_params, shortest_params, longest_params]], columns=["random", "shortest", "longest"], index=[id])


def plot_order(groups: DataFrameGroupBy, name=""):
    fig, axes = set_figure(groups.ngroups)

    df_params = pd.concat([plot_order_routability(ax, id, group, sigmoid) 
                           for ax, (id, group) in zip(axes, groups)])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=[.5, .02, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=len(handles), frameon=False, fontsize="xx-large")
    
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize="xx-large")
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize="xx-large")
    save_fig(f"order/{name}", "lines_points")
    plt.show()
    return df_params

global_mse = 0
def plot_order_routability_modular(ax, id, group, curve_func):
    length = group["length"].values
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    columns = group.columns[group.columns.str.startswith('fraction')]
    df = pd.DataFrame(columns=columns)
    
    print(f"Routability MSE - {id}")
    for i, column in enumerate(columns):
        label = column.replace("fraction_", "")
        c = colors[i%len(colors)]
        # params = plot_curve(ax, length, group[column], curve_func, c, label=label)
        df[column] = [plot_curve(ax, length, group[column], curve_func, c, label=label)]

        global global_mse
        global_mse = get_rmse(length, group[column], curve_func, *df[column]) if get_rmse(length, group[column], curve_func, *df[column]) > global_mse else global_mse
        print(global_mse)
        print(f"  MSE {label:<30} ({id})\n    {get_rmse(length, group[column], curve_func, *df[column])}")

        shape = group["shape"].iloc[0]
        ax.text(.995, .99, f"{shape}", fontsize="x-large", transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    df.index = [id]
    return df
    # return pd.DataFrame([[random_params, shortest_params, longest_params]], columns=["random", "shortest", "longest"], index=[id])


def plot_order_modular(groups: DataFrameGroupBy, name="", cols=None):
    if cols:
        cols = list(groups.obj.columns.intersection(cols))
        groups = groups[cols + ["length", "volume", "shape"]] # not sure if this works
        

    fig, axes = set_figure(groups.ngroups)

    df_params = pd.concat([plot_order_routability_modular(ax, id, group, sigmoid) 
                           for ax, (id, group) in zip(axes, groups)])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=[.5, .02, 0, 0], loc=8,
               fancybox=False, shadow=False, ncol=len(handles) if len(handles) < 6 else len(handles) / 2, frameon=False, fontsize="xx-large")
    
    fig.supylabel("Routability", fontweight="bold",
                  x=0.005, fontsize="xx-large")
    fig.supxlabel("Netlist Length", fontweight="bold", fontsize="xx-large")

    save_fig(f"order/{name}", "lines")
    plt.show()
    return df_params


def order_rpoint(df_params: pd.DataFrame, name="", cols=None):
    if cols:
        df_params = df_params[cols]
    df = df_params.applymap(lambda x: inverse_sigmoid(.5, *x))
    volume = df.index.to_numpy()


    plt.rcParams.update({'font.size': 12})
    kwargs = dict(linestyle="None", marker="o", mew=0, ms=10)
    fig, axes = set_figure(1, figsize=(12, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.11, left=0.06)
    ax: plt.Axes = axes[0]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    print("Rpoint MSE")
    for i, series in enumerate(df):
        ax.plot(volume, df[series], colors[i % len(colors)], **kwargs, label=series)
        params = plot_curve(ax, volume, df[series], log_func, colors[i % len(colors)])
        print(f"  {series:<30}\n    {get_rmse(volume, df[series], log_func, params)}")
    
    def fix_label(label):
        label = label.replace("fraction_", "").replace("adaptive_", "").title()
        parts = label.split("_")
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return " ".join(parts)
        return "{} {} ({})".format(*parts)


    handles, labels = ax.get_legend_handles_labels()
    labels = list(map(fix_label, labels))

    # fig.legend(handles, labels , bbox_to_anchor=[.5, .025, 0, 0], loc=8,
    #         fancybox=False, shadow=False, ncol=len(handles), frameon=False, fontsize="x-large")
    
    fig.legend(handles, labels, loc=4, bbox_to_anchor=(.475, 0.11, .5, .5), ncol=1, fontsize="x-large")
    
    fig.supylabel("Routability Point", fontweight="bold",
                  x=0.005, fontsize="x-large")
    fig.supxlabel("Volume", fontweight="bold", fontsize="x-large")
    ax.set_title("Routability Point against Volume", fontweight="bold", fontsize="xx-large")
    save_fig(f"order/{name}", "order_rpoint")
    plt.show()



import seaborn as sns
def performance_compare_table(df, name="", exclude=[]):
    colnames = df.columns[df.columns.str.startswith('fraction')]
    df = df[df[colnames].eq(df[colnames].iloc[:, 0], axis=0).all(1).apply(lambda x: not x)]
    colnames = df.columns[df.columns.str.startswith('routability')]

    complete_d = {}
    for baseline_name in colnames:
        if baseline_name in exclude:
            continue
        baseline = np.concatenate(df[baseline_name].values)
        bname = baseline_name.replace("routability_", "")
        d = {}
        for compare_name in colnames:
            if compare_name in exclude:
                continue
            test = np.concatenate(df[compare_name].values)
            cname = compare_name.replace("routability_", "")
            d[cname] = sum(baseline < test) / len(baseline) * 100
        complete_d[bname] = d
    df2 = pd.DataFrame(complete_d)
    df2.sort_index(key=df2.sum(1).get, inplace=True, ascending=False)
    df2 = df2[df2.index]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    pal = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, as_cmap=True)
    snsax = sns.heatmap(df2, annot=True, fmt=".2f", linewidths=.5, cbar=False, square=True, cmap=pal, vmax=50, ax=ax,  annot_kws={"size": 18})

    from matplotlib.patches import Rectangle
    snsax.add_patch(Rectangle((0, df2.index.get_loc("random")), len(df2.index), 1, fill=False, edgecolor="darkgreen", lw=1, clip_on=False))
    snsax.add_patch(Rectangle((df2.index.get_loc("random"), 0), 1, len(df2.index), fill=False, edgecolor="darkgreen", lw=1, clip_on=False))
    def change_xlabel(label):
        if label.get_text().strip() == "random":
            label.set_weight("bold")
            return "Random"
        if label.get_text().startswith("adaptive"):
            parts = label.get_text().split("_")
            return f"{parts[1]} {parts[2]}\n({parts[3]})".title()
        
        return label.get_text().replace("_", " ").title()
    
    def change_ylabel(label):
        if label.get_text().strip() == "random":
            label.set_weight("bold")
            return "Random"
        if label.get_text().startswith("adaptive"):
            parts = label.get_text().split("_")
            return f"{parts[1]}\n{parts[2]}\n({parts[3]})".title()
        
        return label.get_text().replace("_", "\n").title()

    ax.set_title("Relative Order Performance", pad=20, fontdict={"fontweight": "bold", "size": 22})
    ax.tick_params(axis='x', rotation=32, labelsize=14)
    ax.tick_params(axis='y', rotation=32, labelsize=14)
    ax.set_ylabel("Better routability than (%)", fontdict={"size": 16}) 
    ax.set_xlabel("Worse routability than (%)", fontdict={"size": 16})
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(list(map(change_xlabel, ax.get_xticklabels())))
    ax.set_yticklabels(list(map(change_ylabel, ax.get_yticklabels())))
    save_fig(f"order/{name}", f"perf_table")
    plt.show()