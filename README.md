# Model Name
<h5 align="center">PEARL</h5>

## Data sets
We provide all data sets (profession data set, hobby data set, and 20News data set [1]) in the folder `data/datasets/`.
It is noted that the profession and hobby data sets are obtained from the authors [2].

## Environment
python=3.6, torch=1.4.0, cudatoolkit=10.0. 
Other packages can be installed via the command `pip install -r requirements.txt`.

## Reproduce
### Preprocess the profession data set via the following commands.

    CUDA_VISIBLE_DEVICES = [gpu_id] python static_representations.py --dataset_name profession
    CUDA_VISIBLE_DEVICES = [gpu_id] python class_oriented_document_representations.py --dataset_name profesion

	Similarly, the hobby (20News) data set can be preprocessed by replacing the profession as hobby(20News).
### Run the framework PEARL on the profession data set via the following command.

    python iterate_frame_profession.py
	
	Similarly, PEARL can run on the hobby (20News) data set via the command "python iterate_frame_hobby.py (iterate_frame_20News.py)"

[1] Lang K. Newsweeder. Learning to filter netnews. Machine Learning Proceedings 1995, 331-339.
[2] Tigunova A, Yates A, Mirza P, et al. CHARM: Inferring personal attributes from conversations. EMNLP'20, 5391-5404.
