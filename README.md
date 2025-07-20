# LDRD_NEAT_ML

## Running the Main ML workflow

Note that the first incantation of the main ML
workflow may take several minutes, but when iterating
or re-running the workflow there are cached operations
that should speed things up (i.e., `pickle` and `joblib`
caching).

Sample incantation: `python main.py --random_seed 42`

## Generating figures for manuscript

Execute the follownig command to generate the figures
used in the manuscript. 

`python -m neat_ml.plot_manuscript_figures`