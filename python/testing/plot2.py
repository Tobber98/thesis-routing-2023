import plotly
import plotly.graph_objects as go
import numpy as np

try:
    from tkinter import Tk
    xres, yres = Tk().winfo_screenwidth() - 100, Tk().winfo_screenheight() - 200
except:
    # Raise exception in log
    xres, yres = 900, 700


class Graph:
    def __init__(self, zlim, ylim, xlim) -> None:
        self.zlim = zlim
        self.ylim = ylim
        self.xlim = xlim

        self.plot_init()

        self.colors = plotly.colors.qualitative.Plotly
        self.ncolors = len(self.colors)

    def plot_init(self):
        self.fig = go.Figure(data=go.Scatter3d())
        self.fig.update_layout(
            scene = dict(
                xaxis = dict(nticks=1, range=[0, self.xlim], backgroundcolor="rgba(0,0,0,0)"),
                yaxis = dict(nticks=1, range=[0, self.ylim], backgroundcolor="rgba(0,0,0,0)"),
                zaxis = dict(nticks=1, range=[0, self.zlim], showgrid=False)
                ),
                width=xres,
                height=yres
            )
        
        X, Y = np.meshgrid([0,self.xlim], [0,self.ylim])
        Z = X*0.
        
        # for z in range(1, self.zlim + 1):
        #     for y in range(0, self.ylim + 1):
        #         self.fig.add_trace(go.Scatter3d(x=(0,self.xlim), y=(y,y), z=(z,z), mode="lines", line=dict(color="grey", width=1), opacity=.1, showlegend=False, hoverinfo="none"))

        #     for x in range(0, self.xlim + 1):
        #         self.fig.add_trace(go.Scatter3d(x=(x,x), y=(0, self.ylim), z=(z,z), mode="lines", line=dict(color="grey", width=1), opacity=.1, showlegend=False, hoverinfo="none"))

        # Change this in a grid to see how that looks
        # self.fig.add_trace(
        #     go.Surface(x=X, y=Y, z=Z + 5, opacity=.1, showscale=False, colorscale=[[0, 'rgb(150,150,150)'], [1, 'rgb(150,150,150)']], hoverinfo='skip')
        # )

    def plot_point(self, coordinates, index):
        z, y, x = coordinates
        self.fig.add_trace(
            go.Scatter3d(
                x=[x], 
                y=[y], 
                z=[z], 
                mode = 'markers', 
                marker=dict(
                    size=4,
                    color=self.colors[index % self.ncolors]
                ), 
                showlegend=False, 
                hoverinfo='none', 
                legendgroup=index
            )
        )

    def plot_line(self, path, index):
        z, y, x = list(zip(*path))
        self.fig.add_trace(
            go.Scatter3d(
                x = x, 
                y=y, 
                z=z, 
                mode="lines", 
                line=dict(
                    color=self.colors[index % self.ncolors], 
                    width=2
                ), 
                legendgroup=index
            )
        ) # check if add traces function can be used to do all paths

    def plot_path(self, path, index):
        self.plot_point(path[0], index)
        self.plot_point(path[-1], index)
        self.plot_line(path, index)

    def plot_paths(self, paths):
        for i, path in enumerate(paths):
            self.plot_path(path, i)

    def show(self):
        self.fig.show()

# g = Graph(10,10,10)
# g.plot_point((1,1,1))
# g.plot_point((3,2,1))
# g.plot_line([(1,1,1), (1,2,1), (2,2,1), (3,2,1)])
# g.show()