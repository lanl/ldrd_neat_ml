# Emulsion Droplets Detection Suite (LANL copyright assertion ID: O# (O4975))

## Installing the project

Download the package from the following GitHub repository:  

```bash
git clone git@github.com:lanl/ldrd_neat_ml.git
```

Install the project, core dependencies,
and optional dependencies by calling:

```
python -m pip install -v ".[dev]" 
```

## Writing a `.yaml` input file for OpenCV or SAM2 detection

The workflow takes as input a `.yaml` configuration file with information
on where to find the input image data for blob detection; save the
output images.

When using the `BubbleSAM` detection method, the `.yaml` file also provides
parameters for mask generation with the `SAM2AutomaticMaskGenerator` function,
postprocessing, and an option to use `cpu` or `gpu` for mask generation.
This `yaml` input file is separate from the `yaml` files used by `SAM-2` to
build the model architecture, e.g. `sam2.1-hiera-l.yaml`, but user provided
mask parameters override the built-in parameters for the `SAM-2` model.

When performing analysis/metric calculation of the resulting bubble detections,
the `yaml` file provides the necessary paths for finding the detection parquet
files; the user generated composition csv file that stores per-image sample
information related to the experimental setup and data collection including
phase separation ground-truth labels and composition weight percentages;
the paths for storing per-image and aggregate metrics. The composition csv
must contain two required columns, 1. "Phase_Separation", which stores the user
provided labels of phase separation status that are used for downstream tasks;
2. "Group", which are unique labels that are used to aggregate per-image metrics
across images that were taken from the same imaging well. The user also provides
a choice of method for calculating graph-based metrics of bubble connectivity
(`knn`, `radius` or `delaunay`). With `graph_method == knn`, the user must provide
a `k_param` integer value denoting the number of nearest neighbors to use for
building the graph. With `graph_method == radius`, the user must provide an
`r_param` integer or float value denoting the radius in pixels to search
for neighboring nodes with which to build the graph. Optionally, the user
can also provide the column names on which to group the aggregate metrics
during analysis. The same per-image and aggregate metrics are calculated for
both detection methods (OpenCV and BubbleSAM).

The `.yaml` file should follow the format below (examples
can be found at `neat_ml/data/opencv_detection_test.yaml`
`neat_ml/data/bubblesam_detection_test.yaml`, and 
`neat_ml/data/opencv_analysis_test.yaml`).
Input paths for `work` and `img_dir` parameters can be
provided as either absolute or relative file paths.

```yaml
roots:
  work: path/to/save/output
  # `results` key is required when providing explicit
  # `analysis:per_image_csv` or `analysis:aggregate_csv` keys
  # or else path generation will fail and throw an error
  results: path/to/save/analysis/outputs

datasets:
  - id: name_of_save_folder
    method: Supports ``OpenCV`` or ``BubbleSAM`` as input
    class: subfolder_for_image_class
    time_label: subfolder_for_timestamp
    # `composition_cols` used for step: analysis
    composition_cols:
      - "Dextran 500 kg/mol (wt%)"
      - "PEO 20 kg/mol (wt%)"
    img_shape: list containing the height and width (in pixels) of the input images, e.g. [2456, 2052] (required for `analysis` step) 

    detection:
      img_dir: path/to/image/data (Can be a directory of ``.tiff`` images or a path to a single ``.tiff`` image.)
      debug: True/False for debug (`True` will save side-by-side figure
             of raw image next to bounding box overlay.)
      # only include the content below when using the ``BubbleSAM`` detection method
      area_threshold: (float) threshold for minimum area (in pixels) of a detected bubble
      circularity_threshold: (float) threshold for minimum circularity of a detected bubble
      # model configuration settings for SAM2
      model_cfg:
        # default mask settings for running the SAM2AutomaticMaskGenerator
        mask_settings:
          points_per_side: 32
          points_per_batch: 128
          pred_iou_thresh: 0.80
          stability_score_thresh: 0.80
          stability_score_offset: 0.10
          crop_n_layers: 4
          box_nms_thresh: 0.10
          crop_n_points_downscale_factor: 1
          min_mask_region_area: 5
          use_m2m: True
        # checkpoint path to download pre-trained SAM-2 weights using ``HuggingFace``
        # see: https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints
        # for list of available checkpoints
        checkpoint_path: "facebook/sam2.1-hiera-large"
        device: "gpu" OR "cpu"
    # only include the content below when calling steps: analysis
    analysis:
      input_dir: path/to/parquet/files
      composition_csv: path/to/composition/information
      per_image_csv: path/to/save/per/image/csv
      aggregate_csv: path/to/save/aggregate/csv 
      group_cols:
        - Group
        - Label
        - Time
        - Class
      graph_method: radius OR knn OR delaunay
      k_param: the number of neighbors to use (int; when using ``graph_method == knn``) 
      r_param: the neighborhood radius in pixels (int, float; when using ``graph_method == radius``)
```

### To run the code on CHICOMA:  
```bash
# before running workflow on back-end node...
# download `SAM2` checkpoints on front-end node using HuggingFace CLI
hf download "facebook/sam2.1-hiera-large"
# load cuda module
module load cudatoolkit/12.6.0
```  

## Detecting Bubbles using SAM-2

> [!IMPORTANT]
> NVIDIA GPU Users: the appropriate `CUDA`, `torch` and `torchvision` versions must be installed for the
> specific GPU on the users system. For download instructions and information visit: 
> https://pytorch.org/get-started/locally/
>
> Measurement of bubbles detected by `SAM2` is optionally handled using the `cucim.skimage.measure` library.
> In order to speed up post-processing of detected bubbles, install the `cucim` package with:
> `python -m pip install cucim-cu12` which requires the `NVIDIA` drivers to be installed first.
> `CUDA` enabled post-processing will only be performed if the user selects `device: "gpu"`
> via the input `yaml` file (as described below) and has a `CUDA` enabled `GPU` available.

The `SAM2` model here uses the `sam2.1_hiera_large.pt` as the checkpoint file to detect
bubbles from microscopy images. The default parameters used for the `SAM2AutomaticMaskGenerator`
are outlined in `neat_ml/data/bubblesam_detection_test.yaml`. These parameters were determined
via visual inspection to increase the number of bubbles detected and with consideration of
computational cost.  

> [!NOTE]
> These parameters were not determined via systematic hyperparameter optimization
> (see: [Issue #13](https://github.com/lanl/ldrd_neat_ml/issues/13))

> [!TIP]
> In order to run the code faster (or with less memory), modify the `points_per_batch` to 64 (from 128).

A description of the parameter settings can be found at:
https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py#L36

> [!WARNING]
> running the workflow using the `MPS` backend on `MacOS` may result in a `NotImplementedError`,
> in which case setting the environment variable `export PYTORCH_ENABLE_MPS_FALLBACK=1` allows `PyTorch`
> to use the `CPU` for unsupported operations on `MPS`.

## Running the OpenCV or SAM2 workflow

To run the workflow with a given `.yaml` file: 

`python run_workflow.py --config <YAML file> --steps detect,analysis (i.e. all)` 

To run the workflow using ``opencv_detection_test.yaml`` (and similarly with ``bubblesam_detection_test.yaml``):

1. download and install the project
2. run the test suite to download the test images from `pooch`
3. run the following command to get the path where the images are stored

```
python -c "import pooch; print(pooch.os_cache('test_images'))"
```

4. replace ``datatsets:detection:img_dir`` `path/to/pooch/images` in the `.yaml` with the local filepath
5. call the `run_workflow` command with `--config neat_ml/data/opencv_detection_test.yaml`

This should process and detect bubbles from the image file `images_raw.tiff` and 
place the outputs under ``roots:work`` filepath from the `.yaml` file

For the `analysis` step, the lines provided in `opencv_analysis_test.yaml` also need to be added to the
input `yaml` file (a description of which can also be found above). These steps process the output bubble
detection data and save 'csv' files containing per-image and aggregated metrics.

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
