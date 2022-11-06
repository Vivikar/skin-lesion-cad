# Skin lesion classification: CAD Project

## 0.1. Set up the environment

Rung following code to set up a conda environment with all the packages needed to run the project.

```
conda update -n base -c defaults conda &&
conda create -n cad_skin anaconda &&
conda activate cad_skin && 
pip install -r requirements.txt
```

## 0.2. Download and set-up the data
Run `skin-lesion-cad/data/make_dataset.py` to download data sets and extract them into corresponding folders
```
python skin-lesion-cad/data/make_dataset.py
```
Or alternatively, download the data sets manually and extract them into corresponding folders.

At the end the following structure of the `/data` folder should be created:
```
data/
├── processed
└── raw
    ├── chall1
    │   ├── train
    │   │   ├── nevus
    │   │   └── others
    │   └── val
    │       ├── nevus
    │       └── others
    └── chall2
        ├── train
        │   ├── bcc
        │   ├── mel
        │   └── scc
        └── val
            ├── bcc
            ├── mel
            └── scc

18 directories
```

## 1. Segment and preprocess the images
Performsbasic preprocessing in images:
* hair removal
* inpainting
* fov artifact detection and removal
* segmentation

Saves inpainted image and mask in corresponding folders in the `/data/processed`.

To run execute:

```$ python -m skin_lesion_cad.utils.segm_script chall1/train/nevus```

Where the parameter passed (`chall1/train/nevus`) defines which images from which folder to process.

A `--resize` option to resize images by a factor can also be passed. For example to downscale image by 2 run.

``` python -m skin_lesion_cad.utils.segm_script chall1/train/nevus --resize 0.5```

So to execute the preprocessing for all the images in the challenge 2 run:
    
    ```
    python -m skin_lesion_cad.utils.segm_script chall2/train/bcc
    python -m skin_lesion_cad.utils.segm_script chall2/train/mel
    python -m skin_lesion_cad.utils.segm_script chall2/train/scc

    python -m skin_lesion_cad.utils.segm_script chall2/val/bcc
    python -m skin_lesion_cad.utils.segm_script chall2/val/mel
    python -m skin_lesion_cad.utils.segm_script chall2/val/scc

    python -m skin_lesion_cad.utils.segm_script chall2/testX
    ```

## 2. Feature Extraction
### 2.1. Extract Keypoint Descriptors from the images (for BoW for challenge 2)

### 2.2. Extract global color and texture features and train BoW

## 3. Model and Feature Selection 

## 4. Ensembling and Predictions