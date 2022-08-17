

import argparse
import copy
import gc
import itertools
import math
import os
import pickle
import time
from collections import defaultdict, Counter

import numpy
import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from preprocessing_utils import load_clean_text, load_tlabels, load_classnames
from tqdm import tqdm

from utils import weight_sentence
from static_representations import handle_sentence
from utils import (INTERMEDIATE_DATA_FOLDER_PATH, MODELS,
                   cosine_similarity_embedding, cosine_similarity_embeddings,
                   evaluate_predictions, tensor_to_numpy, DATA_FOLDER_PATH, ndcg_at_k)


def MinmaxNormalization(mylist):
    if len(mylist) == 1:
        return np.array([1])
    max=np.max(mylist)
    min=np.min(mylist)
    new_list=[]
    for x in mylist:
        new_list.append((x-min)/(max-min))
    return np.array(new_list)


def CountWords_Embeddings(document_statics,document_words,static_word_representations):
    words_id_field = []
    words_embeddings_field=[]
    for doc_ws in document_statics:
        words_id_field.extend(doc_ws)
    words_id_field = list(set(words_id_field))
    for id in words_id_field:
        words_embeddings_field.append(static_word_representations[id])
    return words_id_field,words_embeddings_field



def main(dataset_name, confidence_threshold ,random_state ,lm_type ,layer ,attention_mechanism):
    inter_data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    static_repr_path = os.path.join(inter_data_dir, f"static_repr_lm-bbu.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pickle.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        vocab_words = vocab["vocab_words"]
    with open(os.path.join(inter_data_dir, f"tokenization_lm-bbu.pk"), "rb") as f:
        tokenization_info = pickle.load(f)["tokenization_info"]
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name, f"document_repr_lm-bbu.pk"), "rb") as f:
        dictionary = pickle.load(f)
        document_representations = dictionary["document_representations"]
        class_representations = dictionary["class_representations"]

        static_class_representations = dictionary["static_class_representations"]

        document_statics = dictionary["document_statics"]
        class_words = dictionary["class_words"]

        document_context=dictionary["document_context"]

        document_words = dictionary['document_key_tokens']

        document_word_weights = dictionary['document_tokens_weights']

        document_word_embeddings=dictionary['document_tokens_embeddings']

        document_all_words = dictionary['document_all_tokens']

    epoch = 60

    cur_class_representations=[[class_representations[i]] for i in range(len(class_representations))]

    finished_class = [i for i in range(len(class_representations))]
    finished_document = [i for i in range(len(tokenization_info))]
    exist_words = []
    for i in range(len(class_words)):
        class_words[i] = list(class_words[i])
    for i in range(len(class_words)):
        for j in class_words[i]:
            exist_words.append(j)

    words_id_field,words_embedding_field=CountWords_Embeddings(document_statics,document_words,static_word_representations)
    start = time.time()
    for itra in range(epoch):
        print("iteration num ："+str(itra))
        if len(finished_class) == 0 and len(finished_document) == 0:
            break
        if len(finished_class) == 0:
            print("class stop ！")


        for i in range(len(class_representations)):
            class_representations[i] = np.array(class_representations[i])
        for i in range(len(static_class_representations)):
            static_class_representations[i] = np.array(static_class_representations[i])

        for i in range(len(class_representations)):
            class_representations[i] = np.array(class_representations[i])
        for i in range(len(static_class_representations)):
            static_class_representations[i] = np.array(static_class_representations[i])
        cluster_similarities=[]
        cluster_nearest_words=[]



        for i in range(len(static_class_representations)):
            cluster_similarities.append(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field)))
            cluster_nearest_words.append(np.argsort(-np.array(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field))), axis=1))

        exist_words=[]
        for i in range(len(class_words)):
            for j in class_words[i]:
                exist_words.append(j)


        cur_weights=[[] for i in range(len(class_representations))]

        cur_index = [-1 for i in range(len(class_representations))]
        extended_words = ["" for i in range(len(class_representations))]
        for i in range(len(cluster_nearest_words)):
            if i not in finished_class:
                continue
            new_class_words=cluster_nearest_words[i][0]

            new_index = 0
            for j in range(len(new_class_words)):

                if vocab_words[words_id_field[new_class_words[j]]] not in exist_words:
                    new_index=new_class_words[j]
                    cur_index[i] = new_index

                    extended_words[i]=vocab_words[words_id_field[new_class_words[j]]]
                    class_words[i].append(vocab_words[words_id_field[new_class_words[j]]])
                    exist_words.append(vocab_words[words_id_field[new_class_words[j]]])
                    break
            cur_class_representations[i].append(words_embedding_field[new_index])



        for i in range(len(cur_class_representations)):
            for j in range(len(cur_class_representations[i])):
                cur_weights[i].append(cosine_similarity_embedding(cur_class_representations[i][j],static_class_representations[i]))

        new_class_representations = []
        for i in range(len(cur_class_representations)):
            if i in finished_class:
                new_class_representations.append(
                    np.average(cur_class_representations[i], weights=MinmaxNormalization(cur_weights[i]), axis=0))
            else:
                new_class_representations.append(class_representations[i])

        class_representations = new_class_representations

        if itra <=10 :
            continue

        cluster_similarities = []
        cluster_nearest_words = []

        for i in range(len(static_class_representations)):
            cluster_similarities.append(cosine_similarity_embeddings([class_representations[i]], np.array(words_embedding_field)))
            cluster_nearest_words.append(np.argsort(-np.array(cosine_similarity_embeddings([class_representations[i]], np.array(words_embedding_field))),axis=1))

        for i in range(len(cluster_nearest_words)):
            if i not in finished_class:
                continue
            length = int(len(class_words[i]))
            new_class_words=cluster_nearest_words[i][0][0:length]
            num = 0
            for j in range(len(new_class_words)):

                if vocab_words[words_id_field[new_class_words[j]]] not in class_words[i]:
                    num = num+1
                    if num >= length/4:
                        finished_class.remove(i)
                        cur_class_representations[i].pop()
                        class_words[i].pop()
                        print("finish " + str(i))
                        break
                # else:
                #     if vocab_words[words_id_field[new_class_words[j]]] in class_words[i]:
                #         num = num+1
                #     if num >= length:
                #         exist_words.append(vocab_words[words_id_field[new_class_words[j]]])
                #         class_words[i].append(vocab_words[words_id_field[new_class_words[j]]])
                #         break


        # cluster_similarities = []
        # cluster_nearest_words = []
        #
        # for i in range(len(static_class_representations)):
        #     cluster_similarities.append(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field)))
        #     cluster_nearest_words.append(np.argsort(-np.array(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field))),axis=1))
        # for i in range(len(cluster_nearest_words)):
        #     if i not in finished_class:
        #         continue
        #     length = len(class_words[i])
        #     new_class_words = cluster_nearest_words[i][0]
        #     t = 0
        #     for j in range(len(new_class_words)):
        #         if t >= length:
        #             break
        #         if vocab_words[words_id_field[new_class_words[j]]] in class_words[i]:
        #             t = t+1
        #         elif vocab_words[words_id_field[new_class_words[j]]] in exist_words:
        #             continue
        #         else:
        #             print("finish " + str(i))
        #             finished_class.remove(i)
        #             class_words[i].pop()
        #             cur_class_representations[i].pop()
        #             break
        if itra == 40:
            new_document_words=[]
            new_document_representation=[]
            new_token_embeddings=[]
            new_token_weights=[]
            new_cls_ids=[]
            for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
                if i not in finished_document:

                    new_document_words.append(document_words[i])
                    new_document_representation.append(document_representations[i])
                else:
                    document_representation, tokens,token_embeddings,token_weights,cls_ids= weight_sentence(
                                                              new_class_representations,
                                                              document_context[i],
                                                              document_all_words[i],
                                                              document_statics[i],
                                                              60)
                    new_document_words.append(tokens)
                    new_document_representation.append(document_representation)
                    new_token_embeddings.append(token_embeddings)
                    new_token_weights.append(token_weights)
                    new_cls_ids.append(cls_ids)
                    # if flag == 0:
                    #     finished_document.remove(i)


            document_words = new_document_words
            document_representations = new_document_representation
            document_token_embeddings=new_token_embeddings
            document_token_weights = new_token_weights
            documet_token_cls_ids=new_cls_ids
            break

    class_word_similarity_file = open('../data/datasets/hobby/class_word_similarity.txt', 'w',encoding='utf-8')
    document_keywords_file = open('../data/datasets/hobby/document_keywords.txt', 'w', encoding='utf-8')
    bitem_doc_fre_file = open('../data/datasets/hobby/model/bitem_doc_frequency.txt', 'w', encoding='utf-8')
    doc_to_class_similarity_file = open('../data/datasets/profession/model/doc_to_class_similarity.txt', 'w',encoding='utf-8')
    for document in document_words:
        document_keywords_file.write(' '.join(list(document)))
        document_keywords_file.write('\n')
    document_keywords_file.close()

    os.system('python indexDocs.py ../data/datasets/hobby/document_keywords.txt ../data/datasets/hobby/doc_wids.txt ../data/datasets/hobby/voca.txt')
    all_document_words=[]

    vocab_file = open('../data/datasets/hobby/voca.txt','r')
    vocab_lines = vocab_file.readlines()
    for vocab_line in vocab_lines:
        all_document_words.append(str(vocab_line.strip().split('\t')[1]))
    all_document_word_emb= [[] for j in document_words]
    for i in range(len(document_words)):
        for token in document_words[i]:
            all_document_word_emb[i].append(static_word_representations[word_to_index[token]])


    doc_bitems_embeddings_ids = [[] for i in range(len(tokenization_info))]
    bitem_class_similarity = []
    doc_bitem_similarity = [[] for i in range(len(document_words))]
    all_doc_bitem_fre = [[] for i in range(len(tokenization_info))]

    doc_to_class_similarity=[[] for i in range(len(tokenization_info))]
    t=0
    for document in tqdm(document_words):
        doc_items = [i for i in range(len(document))]
        all_embeddings = []
        if len(doc_items) == 1:
            all_doc_bitem_fre[t].append(str(1))
            bitem_class_similarity.append(np.ones(150))
            doc_bitem_similarity[t].append(np.ones(150))
            t = t + 1
            continue
        for m in itertools.combinations(doc_items,2):
            doc_bitems_embeddings_ids[t].append(list(m))
            weights = [document_token_weights[t][i] for i in list(m)]
            #weights = [0.5 for i in list(m)]
            embeddings = [document_token_embeddings[t][i] for i in list(m)]
            bitem_embedding = np.average(embeddings,weights=weights,axis=0)
            all_embeddings.append(bitem_embedding)
            all_doc_bitem_fre[t].append(str(1))
            bitem_similarities = cosine_similarity_embeddings(np.array([bitem_embedding]),class_representations)


            word_similarity1 = \
            cosine_similarity_embeddings(np.array([document_token_embeddings[t][list(m)[0]]]), class_representations)[0]
            word_similarity2 = \
            cosine_similarity_embeddings(np.array([document_token_embeddings[t][list(m)[1]]]), class_representations)[0]
            word_similarity = word_similarity1 * word_similarity2

            bitem_class_similarity.append((np.array(bitem_similarities[0])))
            #bitem_class_similarity.append((np.array(word_similarity)))
            doc_bitem_similarity[t].append(np.max(bitem_similarities[0]))
            doc_to_class_similarity[t].append(np.array(bitem_similarities[0]))
        #doc_repr = np.average(all_embeddings,axis=0)
        # if len(doc_items) != 1:
        #     doc_similarities = cosine_similarity_embeddings(np.array([doc_repr]), class_representations)
        #     doc_similarities = softmax(doc_similarities)
        #     doc_to_class_similarity.append(list((doc_similarities[0])))
        # else:
        #     doc_similarities = np.array([1])
        #     doc_similarities = softmax(doc_similarities)
        #     doc_to_class_similarity.append(list((doc_similarities)))
        t = t+1
    # t=0
    # for document in document_words:
    #     doc_items = [i for i in range(len(document))]
    #     for m in itertools.combinations(doc_items,2):
    #         doc_bitems_embeddings_ids[t].append(list(m))
    #         embeddings = [static_word_representations[word_to_index[document_words[t][i]]] for i in list(m)]
    #         doc_weights = np.max(cosine_similarity_embeddings(np.array(embeddings),class_representations),axis=1)
    #         bitem_embedding = np.average(np.array(embeddings),weights=doc_weights,axis=0)
    #         bitem_similarities = cosine_similarity_embeddings(np.array([bitem_embedding]),class_representations)
    #         bitem_class_similarity.append(np.array(bitem_similarities[0]))
    #     t = t+1
    bitem_class_similarity = np.array(bitem_class_similarity)
    for t in range(bitem_class_similarity.shape[1]):
        for similarity in bitem_class_similarity[:, t]:
            class_word_similarity_file.write(str(similarity) + ' ')
        class_word_similarity_file.write('\n')
    class_word_similarity_file.close()
    #
    # for i in range(len(doc_to_class_similarity)):
    #     for similarity in doc_to_class_similarity[i]:
    #         doc_to_class_similarity_file.write(str(similarity)+' ')
    #     doc_to_class_similarity_file.write('\n')
    # doc_to_class_similarity_file.close()
    # word_to_class_similarity = cosine_similarity_embeddings(np.array(all_document_word_embeddings),
    #                                                         class_representations)
    # for t in range(len(word_to_class_similarity)):
    #     word_to_class_similarity[t] = MinmaxNormalization(word_to_class_similarity[t])
    # for t in range(word_to_class_similarity.shape[1]):
    #     for similarity in word_to_class_similarity[:, t]:
    #         if similarity < 1e-6:
    #             similarity = 1e-6
    #         class_word_similarity_file.write(str(similarity) + ' ')
    #     class_word_similarity_file.write('\n')
    # class_word_similarity_file.close()

    all_doc_bitems = [[] for i in range(len(tokenization_info))]

    t=0
    for document in document_words:
        for m in itertools.combinations(list(document),2):
            all_doc_bitems[t].append(list(m))
        t = t+1

    bitems_doc_dict = dict()
    for i in range(len(all_doc_bitems)):
        for bitem  in all_doc_bitems[i]:
            if "".join(bitem) in bitems_doc_dict.keys():
                bitems_doc_dict["".join(bitem)].append(i)
            if ("".join(bitem[::-1]) in bitems_doc_dict.keys()) and len(set(bitem)) != 1:
                bitems_doc_dict["".join(bitem[::-1])].append(i)
            if "".join(bitem) not in bitems_doc_dict.keys():
                bitems_doc_dict["".join(bitem)] = []
                bitems_doc_dict["".join(bitem)].append(i)
            if ("".join(bitem[::-1]) not in bitems_doc_dict.keys()) and len(set(bitem)) != 1:
                bitems_doc_dict["".join(bitem[::-1])] = []
                bitems_doc_dict["".join(bitem[::-1])].append(i)


    all_doc_bitem_fre = [[] for i in range(len(tokenization_info))]
    for i in range(len(all_doc_bitems)):
        document_bitems = all_doc_bitems[i]
        for bitem in document_bitems:

            bitem_doc_fre = len([bitem for x in document_bitems if set(x) == set(bitem)])
            bitem_all_fre = len(bitems_doc_dict[''.join(bitem)])
            bitem_fre = float(bitem_doc_fre / bitem_all_fre)
            all_doc_bitem_fre[i].append(bitem_fre)
            #all_doc_bitem_fre[i].append(1)

    for t in range(len(all_doc_bitem_fre)):
        for bitem_fre in all_doc_bitem_fre[t]:
            bitem_doc_fre_file.write(str(bitem_fre)+' ')
        bitem_doc_fre_file.write('\n')
    bitem_doc_fre_file.close()
    # for t in range(len(doc_bitem_similarity)):
    #     for bitem_fre in doc_bitem_similarity[t]:
    #         bitem_doc_fre_file.write(str(bitem_fre) + ' ')
    #     bitem_doc_fre_file.write('\n')
    # bitem_doc_fre_file.close()
    # class_partition = [0 for t in range(150)]
    # for m in range(len(documet_token_cls_ids)):
    #     #     word_id = [0 for t in range(72)]
    #     for id in documet_token_cls_ids[m]:
    #         class_partition[id] = class_partition[id] + 1
    # #     doc_word_to_class.append(np.array(word_id))
    # class_sum = sum(class_partition)
    # class_partition = [str(float(i / class_sum)) for i in class_partition]
    # class_partition_file = open('../data/datasets/hobby/class_partition.txt', 'w', encoding='utf-8')
    # class_partition_file.write(' '.join(class_partition))
    # class_partition_file.close()

    for e in range(20):
        print(os.system('../src/btm est 150 '+str(len(vocab_lines))+' '+'0.33 0.01 100 100 ../data/datasets/hobby/doc_wids.txt ../data/datasets/hobby/model/ ../data/datasets/hobby/class_word_similarity.txt '+str(bitem_class_similarity.shape[0])))
        print(os.system('../src/btm inf sum_b 150 ../data/datasets/hobby/doc_wids.txt ../data/datasets/hobby/model/ ../data/datasets/hobby/model/bitem_doc_frequency.txt'))
        cosine_similarities = np.loadtxt('../data/datasets/hobby/model/k150.pz_d')
        # cosine_similarities = []
        # for doc_bitem in doc_to_class_similarity:
        #     cosine_similarities.append(sum(doc_bitem))

        doc_word_to_class=[]
        for m in range(len(documet_token_cls_ids)):
            word_id = [0 for t in range(150)]
            for id in documet_token_cls_ids[m]:
                word_id[id] = word_id[id] + 1
            doc_word_to_class.append(np.array(word_id))
        #repr_probility = cosine_similarity_embeddings(document_representations, class_representations)
        repr_probility = cosine_similarities
        #repr_probility = cosine_similarity_embeddings(document_representations, class_representations)

        repr_prediction = np.argmax(repr_probility,axis=1)
        class_proportion = np.argmax(np.array(doc_word_to_class),axis=1)
        data_dir = os.path.join(DATA_FOLDER_PATH, dataset_name)
        gold_labels = load_tlabels(data_dir)
        classes = load_classnames(data_dir)
        print("class_num " + str(len(classes)))
        # for i in range(len(gold_labels)):
        #     all_classes = [m for m in documet_token_cls_ids[i]]
        #     counter = Counter(all_classes)
        #     values = dict(counter).keys()
        #     if repr_prediction[i] not in values:
        #         repr_probility[i] = doc_word_to_class[i]
        score = 0
        big_count = 0
        big_MRR = 0
        prof_dict = defaultdict(lambda: [0.0, 0])
        gold_set = set([])
        for i in range(len(gold_labels)):
            index_list = list(np.argsort(-repr_probility[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            ranks = np.zeros(len(class_representations))
            for gold in curr_golds:
                gold_set.add(gold)
                gold_index = index_list.index(gold)
                ranks[gold_index] = 1
            score = score + ndcg_at_k(ranks, 1000)
        print("ndcg")
        print(score / len(gold_labels))
        for i in range(len(gold_labels)):
            index_list = list(np.argsort(-repr_probility[i]))
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            for gold in curr_golds:
                gold_index = index_list.index(gold)
                imrr = 1.0 / (gold_index + 1)
                prof_dict[gold][0] += imrr
                prof_dict[gold][1] += 1
        for prof, stats in prof_dict.items():
            big_count += 1
            big_MRR += float(stats[0] / stats[1])
        print("mrr")
        print(big_MRR / big_count)
        true_num = 0
        merge_num=0
        another_num=0
        another_except_num=0
        true_except_num=0
        another_token_clss_file = open('../data/datasets/hobby/another_token_word_cls.txt','w',encoding='utf-8')
        true_token_clss_file = open('../data/datasets/hobby/true_token_word_cls.txt', 'w', encoding='utf-8')
        false_token_clss_file = open('../data/datasets/hobby/false_token_word_cls.txt', 'w', encoding='utf-8')
        for i in range(len(gold_labels)):
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            all_classes = [m for m in documet_token_cls_ids[i]]
            counter = Counter(all_classes)
            values = dict(counter).keys()
            if repr_prediction[i] not in values:
                repr_prediction[i] = class_proportion[i]
            if repr_prediction[i] in curr_golds:
                true_num = true_num + 1
            if class_proportion[i] in curr_golds:
                another_num = another_num+1
            if repr_prediction[i] in curr_golds and class_proportion[i] in curr_golds:
                merge_num = merge_num+1
            if repr_prediction[i] not in curr_golds and class_proportion[i] in curr_golds:
                another_except_num = another_except_num+1
                gold_classes=[classes[j] for j in curr_golds]
                repr_classes = classes[repr_prediction[i]]
                another_token_clss_file.write(' '.join(gold_classes)+'\n')
                another_token_clss_file.write(repr_classes+'\n')
                for t in range(len(document_words[i])):
                    another_token_clss_file.write(document_words[i][t]+':'+str(classes[documet_token_cls_ids[i][t]])+'   ')
                all_classes = [classes[m] for m in documet_token_cls_ids[i]]
                counter = Counter(all_classes)
                another_token_clss_file.write('\n')
                another_token_clss_file.write(str(counter))
                another_token_clss_file.write('\n')
                another_token_clss_file.write('\n')
            if repr_prediction[i]  in curr_golds and class_proportion[i] not in curr_golds:
                true_except_num = true_except_num + 1
                gold_classes = [classes[j] for j in curr_golds]
                repr_classes = classes[class_proportion[i]]
                true_token_clss_file.write(' '.join(gold_classes) + '\n')
                true_token_clss_file.write(repr_classes + '\n')
                for t in range(len(document_words[i])):
                    true_token_clss_file.write(
                        document_words[i][t] + ':' + str(classes[documet_token_cls_ids[i][t]]) + '   ')
                all_classes = [classes[m] for m in documet_token_cls_ids[i]]
                counter = Counter(all_classes)
                true_token_clss_file.write('\n')
                true_token_clss_file.write(str(counter))
                true_token_clss_file.write('\n')
                true_token_clss_file.write('\n')
            if repr_prediction[i] not in curr_golds and class_proportion[i] not in curr_golds:
                gold_classes = [classes[j] for j in curr_golds]
                repr1_classes = classes[class_proportion[i]]
                repr2_classes = classes[repr_prediction[i]]
                false_token_clss_file.write(' '.join(gold_classes) + '\n')
                false_token_clss_file.write(repr1_classes + '\n')
                false_token_clss_file.write(repr2_classes + '\n')
                for t in range(len(document_words[i])):
                    false_token_clss_file.write(
                        document_words[i][t] + ':' + str(classes[documet_token_cls_ids[i][t]]) + '   ')
                all_classes = [classes[m] for m in documet_token_cls_ids[i]]
                counter = Counter(all_classes)
                false_token_clss_file.write('\n')
                false_token_clss_file.write(str(counter))
                false_token_clss_file.write('\n')
                false_token_clss_file.write('\n')

        print("acc")
        print(float(true_num / len(gold_labels)))
        print("another acc")
        print(float(another_num / len(gold_labels)))
        print("merge acc")
        print(float(merge_num / len(gold_labels)))
        print("another except acc")
        print(float(another_except_num / len(gold_labels)))
        print("true except acc")
        print(float(true_except_num / len(gold_labels)))
        pwz_list = np.loadtxt('../data/datasets/hobby/model/k150.pw_z', dtype=np.float)
        pwz = []
        for t in range(pwz_list.shape[1]):
            pwz.append(pwz_list[:, t])
        bitem_class_similarity = []
        for document in document_words:
            doc_items = [i for i in range(len(document))]
            for m in itertools.combinations(doc_items, 2):
                word_s1 = pwz[all_document_words.index(document[list(m)[0]])]
                word_s2 = pwz[all_document_words.index(document[list(m)[1]])]
                word_similarity = word_s1 * word_s2
                bitem_class_similarity.append(word_similarity)
        class_word_similarity_file = open('../data/datasets/hobby/class_word_similarity.txt', 'w', encoding='utf-8')
        bitem_class_similarity = np.array(bitem_class_similarity)
        for t in range(bitem_class_similarity.shape[1]):
            for similarity in bitem_class_similarity[:, t]:
                class_word_similarity_file.write(str(similarity) + ' ')
            class_word_similarity_file.write('\n')
        class_word_similarity_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="hobby")
    parser.add_argument("--confidence_threshold", default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    ##chenhu
    parser.add_argument("--attention_mechanism", type=str, default="norm_1_2")
    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.confidence_threshold, args.random_state, args.lm_type, args.layer,
         args.attention_mechanism)