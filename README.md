# LDRD_NEAT_ML

## Installation

Download the package from the following GitLab repository:  

```bash
git clone --recurse-submodules ssh://git@lisdi-git.lanl.gov:10022/ldrd_dr_neat/ldrd_neat_ml.git

cd ldrd_neat_ml/sam2

git rev-list -n 1 --before="2024-08-06" main

cd ../../
```

To set up the environment:  

```bash
conda env create -n ldrd_neat_ml "python>=3.10"
conda activate ldrd_neat_ml

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

To set up the environment for Segment Anything-2, refer to `INSTALL.md` in segment-anything-2   
repository and ensure that you have access to GPUs for better performance, when running the script.  

If the following ImportError `ImportError: cannot import name '_C' from 'sam2'` occurs during the run, do the following:  

```bash
cd ./neat_ml/sam2

python setup.py build_ext --inplace
```  

To run the code in CHICOMA:  
```bash
module load cudatoolkit/24.7_12.5  
```  

## Detecting Bubbles using SAM-2

Here, we detect bubbles from microscopic images. We use the `sam2_hiera_large.pt` as the 
checkpoint file. The instructions to download the checkpoint file are provided in 
`download_ckpts.sh` in `neat_ml/sam2/checkpoints` subdirectory. The parameters used for 
detecting the bubbles are given in `bubbleSAM()` method in `neat_ml/bubblesam.py`. 
In order to run the script faster, modify the `points_per_side` to 16.

## Running the OpenCV or BubbleSAM workflow

To run the workflow, user must follow the instructions 
give below. 

You can find the relevant information from:  

`python run_workflow.py --help`

Sample incantation:  
`python run_workflow.py --config <YAML file> --steps detect,analysis,train,infer,explain,plot`

## Running the Main ML workflow

Note that the first incantation of the main ML workflow may take several minutes, 
but when iterating or re-running the workflow there are cached operations 
that should speed things up (i.e., `pickle` and `joblib` caching).

Sample incantation: `python main.py --random_seed 42`  
