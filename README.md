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
## writing a `.yaml` input file for OpenCV detection

The workflow takes as input a `.yaml` configuration file with information
on where to find the input image data for blob detection; save the
output images.

The `.yaml` file should follow the format below (an example
can be found at `neat_ml/data/opencv_detection_test.yaml`):

```
roots:
  work: path/to/save/output

datasets:
  - id: name_of_save_folder
    method: subfolder_for_detection_method
    class: subfolder_for_image_class
    time_label: subfolder_for_timestamp

    detection:
      img_dir: path/to/image/data
      debug: True/False for debug
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
