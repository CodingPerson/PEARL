# coding=UTF-8
import argparse
import os
import pickle as pk
import string
from collections import defaultdict, Counter

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from tqdm import tqdm

from preprocessing_utils import load
from utils import INTERMEDIATE_DATA_FOLDER_PATH, MODELS, tensor_to_numpy

from utils import feature_select

punctuation_map = dict((ord(char), None) for char in string.punctuation)
s = nltk.stem.SnowballStemmer('english')
def prepare_sentence(tokenizer, text):
    # setting for BERT
    model_max_tokens = 512
    has_sos_eos = True
    ######################
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    sliding_window_size = max_tokens // 2

    if not hasattr(prepare_sentence, "sos_id"):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode("", add_special_tokens=True)
        print(prepare_sentence.sos_id, prepare_sentence.eos_id)

    tokenized_text = tokenizer.basic_tokenizer.tokenize(text, never_split=tokenizer.all_special_tokens)
    tokenized_to_id_indicies = []

    tokenids_chunks = []
    tokenids_chunk = []

    for index, token in enumerate(tokenized_text + [None]):
        if token is not None:

            tokens = tokenizer.wordpiece_tokenizer.tokenize(token) ##
        if token is None or len(tokenids_chunk) + len(tokens) > max_tokens:
            tokenids_chunks.append([prepare_sentence.sos_id] + tokenids_chunk + [prepare_sentence.eos_id])
            if sliding_window_size > 0:
                tokenids_chunk = tokenids_chunk[-sliding_window_size:]
            else:
                tokenids_chunk = []
        if token is not None:
            tokenized_to_id_indicies.append((len(tokenids_chunks),
                                             len(tokenids_chunk),
                                             len(tokenids_chunk) + len(tokens)))
            tokenids_chunk.extend(tokenizer.convert_tokens_to_ids(tokens))

    return tokenized_text, tokenized_to_id_indicies, tokenids_chunks


def sentence_encode(tokens_id, model, layer):
    input_ids = torch.tensor([tokens_id], device=model.device)

    with torch.no_grad():
        hidden_states = model(input_ids)
    all_layer_outputs = hidden_states[2]

    layer_embedding = tensor_to_numpy(all_layer_outputs[layer].squeeze(0))[1: -1]
    #layer_embedding = tensor_to_numpy(hidden_states[0].squeeze(0))[1: -1]
    return layer_embedding


def sentence_to_wordtoken_embeddings(layer_embeddings, tokenized_text, tokenized_to_id_indicies):
    word_embeddings = []
    for text, (chunk_index, start_index, end_index) in zip(tokenized_text, tokenized_to_id_indicies):
        word_embeddings.append(np.average(layer_embeddings[chunk_index][start_index: end_index], axis=0))
    assert len(word_embeddings) == len(tokenized_text)
    return np.array(word_embeddings)


def handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies, tokenids_chunks):
    layer_embeddings = [
        sentence_encode(tokenids_chunk, model, layer) for tokenids_chunk in tokenids_chunks
    ]
    word_embeddings = sentence_to_wordtoken_embeddings(layer_embeddings,
                                                       tokenized_text,
                                                       tokenized_to_id_indicies)
    return word_embeddings


def collect_vocab(token_list, representation, vocab):
    assert len(token_list) == len(representation)
    for token, repr in zip(token_list, representation):
        if token not in vocab:
            vocab[token] = []
        vocab[token].append(repr)


def estimate_static(vocab, vocab_min_occurrence):
    static_word_representation = []
    vocab_words = []
    vocab_occurrence = []
    for word, repr_list in tqdm(vocab.items(), total=len(vocab)):
        if len(repr_list) < vocab_min_occurrence:
            continue
        vocab_words.append(word)
        vocab_occurrence.append(len(repr_list))
        static_word_representation.append(np.average(repr_list, axis=0))
    static_word_representation = np.array(static_word_representation)
    # print(f"Saved {len(static_word_representation)}/{len(vocab)} words.")
    return static_word_representation, vocab_words, vocab_occurrence

def stem_count(text):
    l_text = text.lower()
    without_punctuation = l_text.translate(punctuation_map)
    tokens = nltk.word_tokenize(without_punctuation)
    without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    cleaned_text = []
    for i in range(len(without_stopwords)):
        cleaned_text.append(without_stopwords[i])
    return " ".join(cleaned_text)

def main(args):
    dataset = load(args.dataset_name)
    print("Finish reading data")

    data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    if args.lm_type == 'bbu':
        dataset["class_names"] = [x.lower() for x in dataset["class_names"]]

    os.makedirs(data_folder, exist_ok=True)
    with open(os.path.join(data_folder, "dataset.pk"), "wb") as f:
        pk.dump(dataset, f)

    data = dataset["cleaned_text"]
    if args.lm_type == 'bbu':
        data = [x.lower() for x in data]
    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.eval()
    model.cuda()

    tokenization_info = []
    import re
    from collections import Counter
    counts = Counter()
    import string

    for text in tqdm(data):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(tokenizer, text)
        counts.update(word.translate(str.maketrans('','',string.punctuation)) for word in tokenized_text)
        
    del counts['']
    updated_counts = {k: c for k, c in counts.items() if c >= args.vocab_min_occurrence}
    word_rep = {}
    word_count = {}
    clean_texts = []
    print(len(updated_counts.keys()))
    for text in tqdm(data):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(tokenizer, text)
        without_stopwords = [w for w in tokenized_text if not w in stopwords.words('english')]
        clean_texts.append(without_stopwords)
    features = feature_select(clean_texts)[0:len(updated_counts.keys())]
    class_names = dataset["class_names"]
    class_words = []
    for names in class_names:
        splits = names.split(" ")
        for s in splits:
            class_words.append(s)
    for text in tqdm(data):
        tokenized_text, tokenized_to_id_indicies, tokenids_chunks = prepare_sentence(tokenizer, text)
        tokenization_info.append((tokenized_text, tokenized_to_id_indicies, tokenids_chunks))
        contextualized_word_representations = handle_sentence(model, args.layer, tokenized_text,
                                         tokenized_to_id_indicies, tokenids_chunks)
        for i in range(len(tokenized_text)):
          word = tokenized_text[i]
          if word in updated_counts.keys() or word in class_words:
            if word not in word_rep:
              word_rep[word] = 0
              word_count[word] = 0
            word_rep[word] += contextualized_word_representations[i]
            word_count[word] += 1
    word_avg={}
    for k,v in word_rep.items():
      word_avg[k] = word_rep[k]/word_count[k]

    
    vocab_words = list(word_avg.keys())
    static_word_representations = list(word_avg.values())
    vocab_occurrence = list(word_count.values())



    with open(os.path.join(data_folder, "tokenization_lm-bbu.pk"), "wb") as f:
        pk.dump({
            "tokenization_info": tokenization_info,
        }, f, protocol=4)

    with open(os.path.join(data_folder, "static_repr_lm-bbu.pk"), "wb") as f:
        pk.dump({
            "static_word_representations": static_word_representations,
            "vocab_words": vocab_words,
            "word_to_index": {v: k for k, v in enumerate(vocab_words)},
            "vocab_occurrence": vocab_occurrence,
            "tf_idf_vocab":features
        }, f, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--vocab_min_occurrence", type=int, default=2)
    args = parser.parse_args()
    print(vars(args))
    main(args)
