# The Framework PEARL
## Data sets
We provide all data sets (profession data set, hobby data set, and 20News data set [1]) in the folder `data/datasets/`.

It is noted that the profession and hobby data sets are obtained from the authors [2].

## Environment
python=3.6, pytorch=1.4.0, cudatoolkit=10.0. 

Other packages can be installed via the command `pip install -r requirements.txt`.

|  | Profession | Hobby |
| --- | :-------------: | :-------------: |
| Corpus Domain | News | News |
| Attribute Values | 71 | 149 |
| User Utterances | 5747 | 5787 |
| Used By | [CHARM](https://aclanthology.org/2020.emnlp-main.434/)  [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248)| [CHARM](https://aclanthology.org/2020.emnlp-main.434/)  [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248) |

## Reproduce
### Preprocess the profession data set:

    CUDA_VISIBLE_DEVICES = [gpu_id] python static_representations.py --dataset_name profession
    CUDA_VISIBLE_DEVICES = [gpu_id] python utterance_word_representations.py --dataset_name profesion

Similarly, the hobby (resp. 20News) data set can be preprocessed by replacing "profession" as "hobby" (resp. "20News").
### Run PEARL on the profession data set:

    python iterate_frame_profession.py

Similarly, PEARL can run on the hobby (resp. 20News) data set via the command "python iterate_frame_hobby.py" (resp. "python iterate_frame_20News.py").

[1] Lang K. Newsweeder. Learning to filter netnews. Machine Learning Proceedings 1995, 331-339.    

[2] Tigunova A, Yates A, Mirza P, et al. CHARM: Inferring personal attributes from conversations. EMNLP'20, 5391-5404.
