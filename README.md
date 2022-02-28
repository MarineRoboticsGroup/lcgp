# Prioritized, Localizability-Constrained Graph Planning

This repo contains code for path-planning of 2-dimensional range-only
multi-robot networks.

## Getting started

To run some sample code:

``` Bash
conda create -y --name lcgp python=3.9 numpy scipy matplotlib
conda activate lcgp
conda install numba
conda install -c conda-forge cvxpy
conda install -c conda-forge chaospy

# run the sample code from the base directory of this repository
python trial.py
```

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


@author Alan Papalia

@author Nicole Thumma
