import itertools
import math
import operator
import os
import numpy as np
from scipy.special import softmax
import numpy as np
from tqdm import tqdm

linewidth = 200
np.set_printoptions(linewidth=linewidth)
np.set_printoptions(precision=3, suppress=True)

from collections import Counter, defaultdict

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel,RobertaTokenizer

MODELS = {
    'bbc': (BertModel, BertTokenizer, 'bert-base-cased'),
    'bbu': (BertModel, BertTokenizer, 'bert-base-uncased'),
    'roberta':(RobertaModel,RobertaTokenizer,'roberta-base')
}

# all paths can be either absolute or relative to this utils file
DATA_FOLDER_PATH = os.path.join('..', 'data', 'datasets')
INTERMEDIATE_DATA_FOLDER_PATH = os.path.join('..', 'data', 'intermediate_data')
# this is also defined in run_train_text_classifier.sh, make sure to change both when changing.
FINETUNE_MODEL_PATH = os.path.join('..', 'models')

import numpy as np
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
def tensor_to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def dot_product_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b))


def cosine_similarity_embedding(emb_a, emb_b):
    return np.dot(emb_a, emb_b) / np.linalg.norm(emb_a) / np.linalg.norm(emb_b)
def norm_2_similarity(emb_a,emb_b):
    return 0


def pairwise_distances(x, y):
    return cdist(x, y, 'euclidean')


def most_common(L):
    c = Counter(L)
    return c.most_common(1)[0][0]


def  evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    acc = accuracy_score(true_class,predicted_class)
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
        print("acc:"+str(acc))
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }


def feature_select(list_words):

    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1


    word_tf = {}
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())


    doc_num = len(list_words)
    word_idf = {}
    word_doc = defaultdict(int)
    for i in tqdm(doc_frequency):
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))


    word_tf_idf = {}
    for i in tqdm(doc_frequency):
        word_tf_idf[i] = word_tf[i] * word_idf[i]


    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select
def probability_confidence(prob):
    return max(softmax(prob))

def rank_by_norm(embeddings,class_embeddings):
    all_doc_weights = []
    all_doc_id=[]
    for emdb in embeddings:
        diff_embedding = [emdb-cls for cls in class_embeddings]
        diff_arry = np.array(diff_embedding)**2
        doc_to_mulclass_sum = np.sum(diff_arry,axis=1)
        doc_to_mulclass_student_sum = 0
        doc_to_mulclass_weights=[]
        for x in doc_to_mulclass_sum:
            doc_to_mulclass_student_sum += 1.0/(x+1)
        for x in doc_to_mulclass_sum:
            doc_to_mulclass_weights.append((1.0/(x+1)) / doc_to_mulclass_student_sum)
        cuur_weight = np.max(doc_to_mulclass_weights)
        embd_class_id = np.argmax(doc_to_mulclass_weights)
        all_doc_weights.append(cuur_weight)
        all_doc_id.append(embd_class_id)
    return MinmaxNormalization(all_doc_weights),all_doc_weights,all_doc_id

def MinmaxNormalization(mylist):
    max=np.max(mylist)
    min=np.min(mylist)
    new_list=[]
    for x in mylist:
        new_list.append((x-min)/(max-min))
    if len(mylist) == 1:
        return np.array([1])
    return np.array(new_list)

def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(softmax(similarity)) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking


def rank_by_relation(embeddings, class_embeddings):
    #relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    #chenhu
    #relation_score = softmax(relation_score)
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    #relation_ranking = {i: relation_score[i] for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking
def rank_norm_weight(weights,embedding,tokens,raw_weights,keywords_num,static_ids,class_ids):
    ranks=[]
    embeds=[]
    token=[]
    all_weights=[]
    ids=[]
    cls_ids=[]

    for r, i in enumerate(np.argsort(-np.array(weights))):
        if r < keywords_num:
            ranks.append(raw_weights[i])
            #ranks.append(keywords_num-r)
            embeds.append(embedding[i])
            token.append(tokens[i])
            all_weights.append(raw_weights[i])
            ids.append(static_ids[i])
            cls_ids.append(class_ids[i])
    return MinmaxNormalization(ranks),embeds,token,all_weights,ids,cls_ids

def mul(l):
    m = 1
    for x in l:
        m *= x + 1
    return m


def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)


def weights_from_ranking(rankings):
    if len(rankings) == 0:
        assert False
    if type(rankings[0]) == type(0):
        rankings = [rankings]
    rankings_num = len(rankings)
    rankings_len = len(rankings[0])
    assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings))

    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}
    if rankings_num == 1:
        assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
    weights = [0.0] * rankings_len
    for i in range(rankings_len):
        weights[i] = 1. / (total_ranking[i] + 1)
    return weights

def weight_sentence_with_attention(contextualized_word_representations, class_representations,document_all_words,document_statics,
                                   keywords_num):

    norm_weights,all_weights,all_class_ids= rank_by_norm(contextualized_word_representations,class_representations)


    ranks, embeddings, token_words, word_weights, ids,cls_ids = rank_norm_weight(norm_weights,
                                                                             contextualized_word_representations,
                                                                             document_all_words, all_weights,keywords_num,
                                                                              document_statics,all_class_ids)
    #ranks = softmax(ranks)
    document_representation = np.average(embeddings, weights=ranks, axis=0)
    return document_representation,token_words,embeddings,ranks,cls_ids


def weight_sentence(
                    class_representations,
                    document_context,
                    document_all_words,
                    document_statics,
                    keywords_num
                    ):

    document_representation,tokens,embeddings,ranks,cls_ids = weight_sentence_with_attention(document_context,class_representations,document_all_words,
                                                                                                             document_statics,keywords_num)

    return document_representation,tokens,embeddings,ranks,cls_ids
