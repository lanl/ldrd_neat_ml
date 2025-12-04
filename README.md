# LDRD_NEAT_ML (LANL copyright assertion ID: O# (O4909))

## Installing the project

Download the package from the following GitHub repository:  

```bash
git clone git@github.com:lanl/ldrd_neat_ml.git
```

To install the project, clone the repository and install
the package, core dependencies, and optional dependencies
by calling:

```
python -m pip install -v ".[dev]" 

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
    method: opencv
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
