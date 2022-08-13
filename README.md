# Model Name
<h5 align="center">PEARL</h5>

## Data sets
We provide all our data sets in `data/datasets/`, which include profession data set, hobby data set and 20News data set.
## Environment
We used python=3.6, torch-1.4.0, cudatoolkit=10.0. \
Other packages can be installed via `pip install -r requirements.txt`.

## Getting Started
These instructions will get you running the codes of PEARL.
### Pre-processing the data

    CUDA_VISIBLE_DEVICES = [gpu_id] python static_representations.py --dataset_name [dataset_name] \
    CUDA_VISIBLE_DEVICES = [gpu_id] python class_oriented_document_representations.py --dataset_name [dataset_name]

### Run PEARL

    python iterate_frame_profession.py 
    python iterate_frame_hobby.py 
    python iterate_frame_20News.py

#### Run on New Datasets
Our method can be easily applied to new datasets, to do that: \
1. Prepare an utterance corpus `dataset.txt` with attribute values `tlabels.txt` and an attribute value name file `classes.txt` under `data/datasets` folder.

2. Follow the reproduce steps.
