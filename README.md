#  MSc Thesis - The Importance of Order in Routing in 3-Dimensional Grids
## 2023 - VU Amsterdam

This repository contains all relevant code for the MSc Thesis: "The Importance of Order in Routing in 3-Dimensional Grids"


# Setup
To be able to run the experiment it is required that python > 3.11.0 and rustc > 1.69.0 are installed. 
It is advised to create a pyvenv in the python folder of the project.
Additional packages required can be install using:

><code> pip install -r requirements.txt</code>

To build the rust routers on Linux and MacOS run:
><code>./setup.sh</code>

To build for windows run:
><code>.\setup.bat</code>

All routers can be build individually as well by navigating into the folder and running:
><code>carbo b -r</code>

# Running the Experiments
## Replication
...

## Order Method Importance
...

# Recreating Figures
Note that to recreate all of the replication figures it is necessary to have the results of the original experiment by Jansen et al. (2020) as well. The code to acquire those results can be found [here](https://github.com/rlhjansen/Point2PointRoutability). 