# LDRD_NEAT_ML (LANL copyright assertion ID: O# (O4909))

## Installation

Download the package from the following GitLab repository:  

```bash
git clone --recurse-submodules git@github.com:lanl/ldrd_neat_ml.git
```

Install the Segment Anything-2 (SAM2) module, including the pre-trained weight checkpoints:

```bash
# install SAM2
cd neat_ml/sam2
pip install -e ".[notebooks]"
# download model checkpoints
cd checkpoints
# modify the provided bash script to download the appropriate checkpoints
git apply ../../checkpoint.diff
sh download_chkpts.sh
```    

NOTE:
    * refer to ``neat_ml/sam2/INSTALL.sh`` for troubleshooting 
    * the appropriate `CUDA`, `torch` and `torchvision` versions must be installed for the
      specific GPU on the users system.   

### To run the code in CHICOMA:  
```bash
module load cudatoolkit/24.7_12.5  
```  

## Detecting Bubbles using SAM-2

The `SAM2` model here uses the `sam2_hiera_large.pt` as the checkpoint file to detect
bubbles from microscopy images. The parameters used for the `SAM2AutomaticMaskGenerator`
are outlined in `bubbleSAM()` method in `neat_ml/bubblesam.py`. These parameters affect
the performance of the model as well as the computational cost. 
In order to run the script faster (or with less memory), modify the `points_per_side` to 16.

## Running the OpenCV or BubbleSAM workflow

To run the workflow, user must follow the instructions 
given below. 

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
