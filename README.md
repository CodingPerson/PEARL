# The Framework PEARL
## Data sets
We provide all data sets (profession data set, hobby data set, and 20News data set [1]) in the folder `data/datasets/`.

It is noted that the profession and hobby data sets are obtained from the authors [2].

## Environment

python=3.6, pytorch=1.4.0, cudatoolkit=10.0.                                                    
GPU：NVIDIA Geforce GTX 3090   CPU：Inter i9-10900X                                  

Other packages are as follow, which can be installed via the command `pip install -r requirements.txt`.
```
    numpy
    scipy
    tqdm
    scikit-learn
    sentencepiece=0.1.91
    transformers
    tensorboardX
    nltk
    os
    sys
    collections
    itertools
    argparse
    subprocess
    pickle
```
## Datasets
Personal attribute prediction datasets

We follow the same task setting as previous personal attribute prediction papers[2-4], where attribute values are NOT explicitly mentioned in utterances and the given candidate attribute values are ranked based on the underlying semantics of utterances.
|     | Profession | Hobby |
| --- | :-------------: | :-------------: |
| Attribute Values | 71 | 149 |
| User Utterances | 5747 | 5787 |
| Used By | [CHARM](https://aclanthology.org/2020.emnlp-main.434/)  [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248)| [CHARM](https://aclanthology.org/2020.emnlp-main.434/)  [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248) |

Weakly supervised text classification dataset

PEARL is also tested on weakly supervised text classification task to verify its universality, flexibility and effectiveness.
| | 20News |
| --- | :-------------: |
| Classes | 5 |
| Documents | 17871 |
| Used By | [X-Class](https://arxiv.org/abs/2010.12794)|
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

[3] Liu Y, Chen H, Shen W. Personal Attribute Prediction from Conversations. WWW'2022, 223-227.

[4] Tigunova A, Yates A, Mirza P, et al. Listening between the lines: Learning personal attributes from conversations. WWW'2019, 1818-1828.

