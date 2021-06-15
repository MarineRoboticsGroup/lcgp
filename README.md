# Rigidity Based Planning

This repo contains code for path-planning of 2-dimensional range-only
multi-robot networks.

## Getting started

To run some sample code:

``` Bash
# setup the environment
conda create -y --name rigidity_planning python=3.6 numpy scipy matplotlib
conda activate rigidity_planning
conda install numba
pip install -r requirements.txt

# faiss install depends on hardware available
# see link for further detail: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
conda install -c pytorch faiss-gpu cudatoolkit=10.2

# run the sample code from the base directory of this repository
python trial.py
```

## Files and Folders

- trial.py: main script from which path planning experiments are run
- planning related classes
  - swarm.py: class to represent the network of robots and their current state
  - graph.py: class to represent the spatially embedded graph formed by a network of robots
  - environment.py: class to represent the environment to plan in
  - kdtree.py: class to build kdtree for efficient spatial querying
  - planners/: classes of different planners
  - kdtree.py: class to build kdtree for efficient spatial querying
- utils
  - math_utils.py: general utils for calculations
  - plot.py: utils for plotting
- test_scripts/
  - test_eigval_computations.py: test fastest eigenvalue computations
  - test_fisher_matrix_computations.py: test fastest way of constructing FIM
  - test_rigidity_snl.py: build plots to compare SNL error to rigidity value of matrix
  - test_configs.py: test hand-made configurations of networks to test ideas on
    rigidity (includes interactive click to place node feature)
  - test_cache_rigidity.py: test idea of caching rigidity values. Relates to rigidity_library.py
  - test_binary_matrices.py: test more efficient ways of representing the
    locations of robots on grid
- not used
  - rigidity_library.py: class meant to hold pre-recorded rigidity values to
    avoid eigenvalue computations

## Code Profiling

to perform profiling you can just set `profile=True` in the main function of
*trial.py* and a script will be called that does all of the post-processing for
you. To see how this script works you can look at *profiling/flamegraph.bash*

@author Alan Papalia
