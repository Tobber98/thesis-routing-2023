from src.plotter.plotter import PlotlyGraph, MLPGraph

# plotter = PlotlyGraph(1, 7, 7)
plotter = MLPGraph(1, 7 ,7)
# plotter.fig.update_layout(scene = dict(zaxis=dict(nticks=1)))

# Expensive route
# plotter.plot_paths([[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1)], [(0, 1, 3), (0, 2, 3), (0, 3, 3), (0, 4, 3), (0, 5, 3)], [(0, 3, 2), (0, 4, 2), (0, 5, 2), (0, 6, 2), (0, 6, 3), (0, 6, 4), (0, 5, 4), (0, 4, 4)]])

# # Better route
# plotter.plot_paths([[(0, 1, 1), (0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 5, 1)], [(0, 1, 3), (0, 2, 3), (0, 2, 4), (0, 2, 5), (0, 3, 5), (0, 4, 5), (0, 4, 4), (0, 4, 3), (0, 5, 3)], [(0, 3, 2), (0, 3, 3), (0, 3, 4)]])

# Bad route
# plotter.plot_paths([[(0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 4, 2), (0, 4, 3)], [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 4), (0, 3, 4), (0, 4, 4)]])
# plotter.plot_point((0, 3, 2), "blue")
# plotter.plot_point((0, 3, 5), "blue")

# Good route
plotter.plot_paths([[(0, 2, 1), (0, 3, 1), (0, 4, 1), (0, 4, 2), (0, 4, 3)], [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 1, 6), (0, 2, 6), (0, 3, 6), (0, 4, 6), (0, 4, 5), (0, 4, 4)], [(0, 3, 2), (0, 3, 3), (0, 3, 4), (0, 3, 5)]])
plotter.show()