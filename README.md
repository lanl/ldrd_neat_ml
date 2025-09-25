# LDRD_NEAT_ML (LANL copyright assertion ID: O# (O4909))

## Installation

Download the package from the following GitLab repository:  

```bash
git clone ssh://git@lisdi-git.lanl.gov:10022/ldrd_dr_neat/ldrd_neat_ml.git

```

To set up the environment:  

```bash

cd ldrd_neat_ml
conda env create -n ldrd_neat_ml "python>=3.10"
conda activate ldrd_neat_ml

pip install -r requirements.txt
```

## Running the OpenCV detection

To run the workflow, user must follow the instructions 
give below. 

You can find the relevant information from:  

`python run_workflow.py --help`

Sample incantation: `python run_workflow.py --config <YAML file> --steps detect`

## Running the Main ML workflow

Note that the first incantation of the main ML
workflow may take several minutes, but when iterating
or re-running the workflow there are cached operations
that should speed things up (i.e., `pickle` and `joblib`
caching).

Sample incantation: `python main.py --random_seed 42`
