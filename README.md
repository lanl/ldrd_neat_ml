# LDRD_NEAT_ML (LANL copyright assertion ID: O# (O4975))

## Installing the project

Download the package from the following GitHub repository:  

```bash
git clone --recurse-submodules git@github.com:lanl/ldrd_neat_ml.git
```

Install the Segment Anything-2 (SAM2) module, including the pre-trained weight checkpoints:

```bash
# install SAM2
cd neat_ml/sam2
pip install -e
# download model checkpoints
cd checkpoints
# modify the provided bash script to download the appropriate checkpoints
git apply ../../download_checkpoints.diff
sh download_ckpts.sh
```    

NOTE:
* refer to ``neat_ml/sam2/INSTALL.sh`` for troubleshooting 
* GPU Users: the appropriate `CUDA`, `torch` and `torchvision` versions must be installed for the
specific GPU on the users system. For download instructions and information visit:

https://pytorch.org/get-started/locally/

Install the project, core dependencies,
and optional dependencies by calling:

```
python -m pip install -v ".[dev]" 
```


### Running the test-suite

After installation, the test-suite can be run using:

```
pyton -m pytest
```

## Writing a `.yaml` input file for OpenCV or SAM2 detection

The workflow takes as input a `.yaml` configuration file with information
on where to find the input image data for blob detection; save the
output images.

The `.yaml` file should follow the format below (examples
can be found at `neat_ml/data/opencv_detection_test.yaml`
and `neat_ml/data/bubblesam_detection_test.yaml`)
Input paths can be provided as either absolute or relative
file paths.

```yaml
roots:
  work: path/to/save/output

datasets:
  - id: name_of_save_folder
    method: Supports ``OpenCV`` or ``BubbleSAM`` as input
    class: subfolder_for_image_class
    time_label: subfolder_for_timestamp

    detection:
      img_dir: path/to/image/data (Can be a directory of ``.tiff`` images or a path to a single ``.tiff`` image.)
      debug: True/False for debug (`True` will save side-by-side figure
             of raw image next to bounding box overlay.)
```

### To run the code in CHICOMA:  
```bash
module load cudatoolkit/24.7_12.5  
```  

## Detecting Bubbles using SAM-2

The `SAM2` model here uses the `sam2_hiera_large.pt` as the checkpoint file to detect
bubbles from microscopy images. The parameters used for the `SAM2AutomaticMaskGenerator`
are outlined in `DEFAULT_MASK_SETTINGS` dictionary in `neat_ml/bubblesam/bubblesam.py`.
These parameters affect the performance of the model as well as the computational cost. 
In order to run the script faster (or with less memory), modify the `points_per_side` to 16.
A description of the parameter settings can be found at:

https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L36

## Running the OpenCV or SAM2 workflow

To run the workflow with a given `.yaml` file: 

`python run_workflow.py --config <YAML file> --steps detect`

To run the workflow using ``opencv_detection_test.yaml``:

1. download and install the project
2. run the test-suite to download the test images from `pooch`
3. run the following command to get the path where the images are stored

```
python -c "import pooch; print(pooch.os_cache('test_images'))"
```

4. replace ``datatsets:detection:img_dir`` `path/to/pooch/images` in the `.yaml` with the local filepath
5. call the `run_workflow` command with `--config neat_ml/data/opencv_detection_test.yaml`

This should process and detect bubbles from the image file `images_raw.tiff` and 
place the outputs under ``roots:work`` filepath from the `.yaml` file

For information relevant to running the workflow:  

`python run_workflow.py --help`

## Running the Main ML workflow

Note that the first incantation of the main ML
workflow may take several minutes, but when iterating
or re-running the workflow there are cached operations
that should speed things up (i.e., `pickle` and `joblib`
caching).

Sample incantation: `python main.py --random_seed 42`

## Generating figures for manuscript

### Downloading Arial Typeface (Debian based Linux Users)

Plots use Arial typeface, which is available on Windows
and Mac systems by default, but needs to be downloaded
and installed for Linux based systems. Plot generation
will run without the font, but the test-suite will not pass.
To download this font (for Debian based linux systems)
run the following commands:

```
echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | sudo debconf-set-selections
sudo apt-get install -y ttf-mscorefonts-installer || true
sudo fc-cache -fv
```

Alternatively, you can download the fonts directly and
install them with the following commands:

```
sudo apt-get update
sudo apt-get install -y cabextract
sudo apt-get install -y xfonts-utils
wget http://ftp.de.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.8.1_all.deb
sudo dpkg -i ttf-mscorefonts-installer_3.8.1_all.deb || true
sudo fc-cache -fv
```

### Plot generation

After installing the appropriate font libraries, execute
the following command to generate the figures used in the
manuscript: 

`python -m neat_ml.plot_manuscript_figures`
