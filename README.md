# Prioritized, Localizability-Constrained Graph Planning

This repo contains code for path-planning of 2-dimensional range-only
multi-robot networks.

## Getting started

To run some sample code:

``` Bash
# setup the environment from the file provided
conda create --name rigidity_planning --file requirements.txt

# alternatively, setup from scratch
conda create -y --name rigidity_planning python=3.9 numpy scipy matplotlib
conda activate rigidity_planning
conda install numba
conda install -c conda-forge cvxpy
conda install -c conda-forge chaospy
pip install git+https://github.com/evanhempel/python-flamegraph.git

# run the sample code from the base directory of this repository
python trial.py
```

## Potential Issues

While not as elegant as we would like, due to the implementation there may be
need for some parameter tuning. Here we list some issues where the various
parameters of the planner may benefit from tuning.

If the planner is finding conflicts and the message being printed is `valid set
not growing`, this may be due to one of two issues. As a first 

## Useful for debugging

If the planner is having issues planning and it is unclear if it's a bug or if
it genuinely is unable to find paths, inside the `prioritized_prm.py` file we
have the ability to plot the trajectories for all robots `[0,..,n-1]` and show the
state of all of the points on the PRM (which set they belong to, if any) of
robot `n`. As default there is a `if False:` branch in the
`perform_planning` function, to turn this plotting on change `False` to `True`.
This will plot after each robot plans its path (remember, prioritized planning
so the robots plan independently in sequence).
The full code looks like this:

```Python
if False:
    self.constraintSets.animate_valid_states(
        self._coord_trajs, cur_robot_id + 1
    )
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

@author Nicole Thumma
