# The Framework PEARL
## Environment

Computational platform: PyTorch 1.4.0, NVIDIA Geforce GTX 3090 (GPU), Inter i9-10900X (CPU), CUDA Toolkit 10.0

Development language: Python 3.6/C++
       
Liabraries are listed as follow, which can be installed via the command `pip install -r requirements.txt`.
```
numpy, scipy, tqdm, scikit-learn, sentencepiece=0.1.91, transformers, tensorboardX, nltk, os, sys, collections, itertools, argparse, subprocess, pickle, cudatoolkit=10.0, pytorch==1.4.0
```
## Data sets
We provide all the data sets (profession data set, hobby data set, and 20News data set) in the folder `data/datasets/`. 
### Personal attribute prediction task
Profession data set(obtained from the authors of [2])
atribute values: 71; user utterances: 5747 
used by the previous work: [CHARM](https://aclanthology.org/2020.emnlp-main.434/) [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248) 

Hobby data set (obtained from the authors of [2])
atribute values: 149; user utterances: 5787
used by the previous work: [CHARM](https://aclanthology.org/2020.emnlp-main.434/) [DSCGN](https://dl.acm.org/doi/abs/10.1145/3487553.3524248) 

Note that we follow the same task setting as previous personal attribute prediction papers[2-4], where attribute values are NOT explicitly mentioned in utterances and the given candidate attribute values are ranked based on the underlying semantics of utterances.

### Weakly supervised text classification task
20News data set(obtained from [1])
classes: 5; documents: 17871    
used by the previous work: [X-Class](https://arxiv.org/abs/2010.12794) 

Note that PEARL is tested on the weakly supervised text classification task to verify its universality, flexibility and effectiveness.

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

