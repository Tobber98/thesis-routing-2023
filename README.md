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
To run the replication experiment, all that needs to be done is to run the run_replication.py script with python with the <code>-r</code> command. Fully:
><code>python run_replication.py -r</code>

## Order Method Importance
To run the order router experiment use:
><code>python run_order_test.py</code>

# Analysis and Recreating Figures
To recreate the figures first the results have to be gathered by running the experiments as explained above.
## Replication
Note that to recreate all of the replication figures it is necessary to have the results of the original experiment by Jansen et al. (2020) as well. The code to acquire those results can be found [here](https://github.com/rlhjansen/Point2PointRoutability). 

The figures for the replication can be created by using:
><code>python run_replication.py -a </code>

## Order Method Importance
The figures for the order experiment can be created by using:
><code>python analyse_order_results.py</code>