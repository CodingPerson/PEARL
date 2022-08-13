import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import tqdm

from static_representations import handle_sentence
from utils import (INTERMEDIATE_DATA_FOLDER_PATH, MODELS,
                   cosine_similarity_embedding, cosine_similarity_embeddings,
                   evaluate_predictions, tensor_to_numpy)


def probability_confidence(prob):
    return max(softmax(prob))

def rank_by_norm(embeddings,class_embeddings):
    all_doc_weights = []
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
        all_doc_weights.append(cuur_weight)
    return MinmaxNormalization(all_doc_weights),all_doc_weights

def MinmaxNormalization(mylist):
    max=np.max(mylist)
    min=np.min(mylist)
    new_list=[]
    for x in mylist:
        new_list.append((x-min)/(max-min))
    return np.array(new_list)

def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(softmax(similarity)) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking


def rank_by_relation(embeddings, class_embeddings):
    relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking
def rank_norm_weight(weights,embedding,tokens,raw_weights,keywords_num,static_ids):
    ranks=[]
    embeds=[]
    token=[]
    all_weights=[]
    ids=[]
    for r, i in enumerate(np.argsort(-np.array(weights))):
        ranks.append(weights[i])
        embeds.append(embedding[i])
        token.append(tokens[i])
        all_weights.append(raw_weights[i])
        ids.append(static_ids[i])
    return ranks[0:keywords_num],embeds[0:keywords_num],token[0:keywords_num],all_weights[0:keywords_num],ids[0:keywords_num]

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

def weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations, class_representations,
                                   attention_mechanism,keywords_num):
    assert len(tokenized_text) == len(contextualized_word_representations)

    contextualized_representations = []
    static_representations = []
    static_ids=[]
    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    tf_idf_vocab = vocab["tf_idf_vocab"]
    tf_words = []
    tokens=[]
    for tf in tf_idf_vocab:
        tf_words.append(tf[0])
    for i, token in enumerate(tokenized_text):
        if token in word_to_index and token in tf_words:
            static_representations.append(static_word_representations[word_to_index[token]])
            contextualized_representations.append(contextualized_word_representations[i])
            static_ids.append(word_to_index[token])
            tokens.append(token)
    if len(contextualized_representations) == 0:
        print("Empty Sentence (or sentence with no words that have enough frequency)")
        static_representations.append(static_word_representations[word_to_index['word']])
        contextualized_representations.append(static_word_representations[word_to_index['word']])
        static_ids.append(word_to_index['word'])
        tokens.append('word')


    norm_weights,all_weights = rank_by_norm(contextualized_representations,class_representations)
    ranks,embeddings,token_words,word_weights,ids= rank_norm_weight(norm_weights,contextualized_representations,tokens,all_weights,keywords_num,static_ids)

    return np.average(embeddings, weights=ranks, axis=0),static_ids,contextualized_representations,token_words,tokens,ranks,embeddings


def weight_sentence(model,
                    vocab,
                    tokenization_info,
                    class_representations,
                    attention_mechanism,
                    layer,
                    keywords_num
                    ):
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = tokenization_info
    contextualized_word_representations = handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies,
                                                          tokenids_chunks)

    document_representation,static_ids,context_embeddings,tokens,all_tokens,token_weights,embeddings = weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations,class_representations,
                                                                                                        attention_mechanism,keywords_num)
    return document_representation,static_ids,context_embeddings,tokens,all_tokens,token_weights,embeddings


def main(args):
    data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    with open(os.path.join(data_folder, "dataset.pk"), "rb") as f:
        dataset = pk.load(f)
        class_names = dataset["class_names"]

    static_repr_path = os.path.join(data_folder, 'static_repr_lm-bbu.pk')
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        vocab_words = vocab["vocab_words"]
        tf_idf = vocab["tf_idf_vocab"]
    with open(os.path.join(data_folder, "tokenization_lm-bbu.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]
    tf_words = []
    for tf in tf_idf:
        tf_words.append(tf[0])
    print("Finish reading data")

    print(class_names)

    masked_words = set()
    for cls in range(len(class_names)):
        split_words = class_names[cls].split(" ")
        for split in split_words:
            masked_words.add(split)
    class_words = [class_names[cls].split(" ") for cls in range(len(class_names))]
    all_class_words=[]
    for i  in range(len(class_words)):
        for j in class_words[i]:
            all_class_words.append(j)
    class_words_representations = []
    class_repres=[]
    for cls in range(len(class_names)):
        if len(class_names[cls].split(" ")) != 1:
            split_words = class_names[cls].split(" ")
            word = split_words[0]
            word_representattion = np.zeros_like(static_word_representations[word_to_index[word]])
            for split in split_words:
                word_representattion = np.sum([static_word_representations[word_to_index[split]],word_representattion],axis=0)
            word_representattion = [i / len(split_words) for i in word_representattion]
        else:
            word=class_names[cls]
            word_representattion = static_word_representations[word_to_index[word]]
        class_words_representations.append([word_representattion])
        class_repres.append(word_representattion)
    print(np.array(class_words_representations).shape)



    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.eval()
    model.cuda()

    document_representations = []
    document_statics=[]
    document_context=[]
    document_key_tokens=[]

    document_all_tokens=[]
    document_key_tokens_weights=[]
    document_key_tokens_embeddings = []
    for _tokenization_info in tqdm(tokenization_info):
        document_representation,static_ids,context_embeddings,tokens,all_tokens,weights,embeddings= weight_sentence(model,
                                                  vocab,
                                                  _tokenization_info,
                                                  np.array(class_repres),
                                                  args.attention_mechanism,
                                                  args.layer,2)
        document_representations.append(document_representation)
        document_statics.append(static_ids)
        document_context.append(context_embeddings)
        document_key_tokens.append(tokens)
        document_all_tokens.append(all_tokens)
        document_key_tokens_weights.append(weights)
        document_key_tokens_embeddings.append(embeddings)
    document_representations = np.array(document_representations)

    print("Finish getting document representations")
    with open(os.path.join(data_folder,
                           f"document_repr_lm-{args.lm_type}.pk"),
              "wb") as f:
        pk.dump({
            "static_class_representations":class_repres,
            "class_words": class_words,
            "class_representations": np.array(class_repres),
            "document_representations": document_representations,
            "document_statics":document_statics,
            "document_context":document_context,
            "document_key_tokens":document_key_tokens,
            "document_all_tokens":document_all_tokens,
            "document_tokens_weights":document_key_tokens_weights,
            "document_tokens_embeddings":document_key_tokens_embeddings
        }, f, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--attention_mechanism", type=str, default="iterate_1_1")
    args = parser.parse_args()
    print(vars(args))
    main(args)
