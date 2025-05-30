
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
from utils.index_utils import extract_texts, extract_sentences_from_texts, encode_sentences
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import pandas as pd
import sqlite3
import random
import argparse



def main(argv):
    H = FLAGS.config

    device = H.run.device if torch.cuda.is_available() else -1
    os.makedirs("random_sentences", exist_ok=True)
    if H.model.type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
        model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)
        model.resize_token_embeddings(len(tokenizer))
        mask_filler = pipeline("fill-mask", model = H.model.model, tokenizer = H.model.tokenizer, top_k = H.model.top_k, device=device)

    elif H.model.type == "sentence-transformers":
        model = SentenceTransformer(H.model.model).to(device)
        tokenizer = None

    elif H.model.type == "bert":
        tokenizer = BertTokenizer.from_pretrained(H.model.tokenizer)
        model = BertForMaskedLM.from_pretrained(H.model.model).to(device)

    elif H.model.type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(H.model.model).get_encoder().to(device)

    #path to json dataset
    json_dataset_path = H.data.json_dataset_path


    num_current_folder = 1
    all_sentences = []
    for folder_name in os.listdir(json_dataset_path):
        folder_path = os.path.join(json_dataset_path, folder_name)

        # check if  is a directory
        if os.path.isdir(folder_path):
            print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

            # if not (folder_name == "Himerius Soph. (2051)"):
            #      continue
            sentence_per_author = []
            # Iterate on each json file
            for file_name in os.listdir(folder_path)[:1]:
                file_path = os.path.join(folder_path, file_name)
                # Check if is a JSON file
                if file_name.endswith(".json"):
                    print(f"    JSON: {file_name}")
                    with open(file_path, "r", encoding="utf-8") as json_file:

                        data = json.load(json_file)
                        try:
                            #extract all text fields
                            texts, citations = extract_texts(data)
                            #join texts into a single sentence if they semantically belong together
                            sentences, citations = extract_sentences_from_texts(texts, citations, mask_filler, H.data.min_words_in_phrase, H.model.model_max_length, tokenizer )
                            sentence_per_author.extend(sentences)
                        except Exception as e:
                            print(f"Error in {file_name}: {e}")
                            continue
            #add sentences to all sentences
            num_samples = len(sentence_per_author)
            if num_samples < 250:
                all_sentences.extend(sentence_per_author)
            else:
                all_sentences.extend(random.sample(sentence_per_author, 250))
            num_current_folder += 1

    print(f"Total number of sentences: {len(all_sentences)}")
    sentences = random.sample(all_sentences, 1000)

    for i in range(120):
        x = open(f"random_sentences/random_sentences_{i}.txt", "w")
        for sentence in sentences:
            x.write(sentence + "\n")
        x.close()


if __name__ == '__main__':
    # Commandline arguments
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
    flags.mark_flags_as_required(["config"])
    app.run(main)