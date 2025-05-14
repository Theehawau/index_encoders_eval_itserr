
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
import faiss
import pandas as pd
import sqlite3
import random
import argparse





def main(argv):
    H = FLAGS.config
    H.model.tokenizer = FLAGS.tokenizer
    H.model.model = FLAGS.model
    H.model.type = FLAGS.type

    H.index.index_name = H.index.index_name.format(FLAGS.index_model)
    H.index.idx_2_keys = H.index.idx_2_keys.format(FLAGS.index_model)

    H.db.db_name = H.db.db_name.format(FLAGS.index_model)

    device = H.run.device if torch.cuda.is_available() else -1

    #path to json dataset
    json_dataset_path = H.data.json_dataset_path

    if H.model.type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
        model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)
        model.resize_token_embeddings(len(tokenizer))
        mask_filler = pipeline("fill-mask", model = H.model.model, tokenizer = H.model.tokenizer, top_k = H.model.top_k, device=device)

    elif H.model.type == "sentence-transformers":
        model = SentenceTransformer(H.model.model).to(device)

    elif H.model.type == "bert":
        tokenizer = BertTokenizer.from_pretrained(H.model.tokenizer)
        model = BertForMaskedLM.from_pretrained(H.model.model).to(device)

    elif H.model.type == "t5":
        tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(H.model.model).get_encoder().to(device)


    #create index
    path_to_save_index = os.path.join(H.index.index_path,f"{H.index.index_name}.index")
    # Controlla se il file dell'indice esiste già
    if os.path.exists(path_to_save_index):
        # Carica l'indice esistente
        index = faiss.read_index(path_to_save_index)
        print("Index successfully loaded!")
    else:
        # Se non esiste, crea un nuovo indice
        index = faiss.IndexFlatL2(H.data.len_embedding)
        print("New index created.")
    

    #create or open db
    path_to_save_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_save_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot create or open database."

    cursor = connection.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {H.db.db_name} (row_id INTEGER PRIMARY KEY AUTOINCREMENT, sentence TEXT)")



    # num_current_folder = 1   
    # all_sentences = []
    # for folder_name in os.listdir(json_dataset_path):
    #     folder_path = os.path.join(json_dataset_path, folder_name)

    #     # check if  is a directory
    #     if os.path.isdir(folder_path):
    #         print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

    #         # if not (folder_name == "Himerius Soph. (2051)"):
    #         #      continue
    #         sentence_per_author = []
    #         # Iterate on each json file
    #         for file_name in os.listdir(folder_path)[:1]:
    #             file_path = os.path.join(folder_path, file_name)
    #             # Check if is a JSON file
    #             if file_name.endswith(".json"):
    #                 print(f"    JSON: {file_name}")
    #                 with open(file_path, "r", encoding="utf-8") as json_file:

    #                     data = json.load(json_file)
    #                     try:
    #                         #extract all text fields
    #                         texts, citations = extract_texts(data)
    #                         #join texts into a single sentence if they semantically belong together
    #                         sentences, citations = extract_sentences_from_texts(texts, citations, mask_filler, H.data.min_words_in_phrase, H.model.model_max_length, tokenizer )
    #                         sentence_per_author.extend(sentences)
    #                     except Exception as e:
    #                         print(f"Error in {file_name}: {e}")
    #                         continue
    #         #add sentences to all sentences
    #         num_samples = len(sentence_per_author)
    #         if num_samples < 250:
    #             all_sentences.extend(sentence_per_author)
    #         else:
    #             all_sentences.extend(random.sample(sentence_per_author, 250))
    #         num_current_folder += 1
    
    # print(f"Total number of sentences: {len(all_sentences)}")
    # sentences = random.sample(all_sentences, 1000)
    # print(f"Number of sentences to index: {len(sentences)}")

    # x = open("random_sentences.txt", "w")
    # for sentence in sentences:
    #     x.write(sentence + "\n")
    # x.close()
    sentences = []
    with open("random_sentences.txt", "r") as f:
        for line in f:
            sentences.append(line.strip())

    df = pd.read_csv(json_dataset_path, sep="\t", encoding="utf-8")
    
    targets = [f"Target #{i}" for i in range(1, 6)] 

    for i, target in enumerate(targets):
         sentences.extend(df[target].values.tolist())


    print(f"Number of sentences to index: {len(sentences)}")
    
        
    #get encoding of each sentence
    sentence_embeddings = encode_sentences(sentences, model, tokenizer, H.data.len_embedding, device, H.model.model_max_length)

    #add embeddings to index
    faiss.normalize_L2(sentence_embeddings)
    index.add(sentence_embeddings)
    
    #save data to db (NB: if FAISS USE INDEX K FOR A SENTENCE, SQLITE USE INDEX (K+1))
    for idx, sentence in enumerate(sentences):
        cursor.execute(f"""
                            INSERT INTO {H.db.db_name} (sentence) 
                            VALUES (?)
                        """, (sentence,))
    connection.commit()
    #save index
    faiss.write_index(index, path_to_save_index)


    connection.close()



if __name__ == '__main__':
    config_name = "config/target_best1k_index_config.py"
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        type=str, 
                        default="bowphs/LaTa", 
                        help="Model to use for the search")
    parser.add_argument("--tokenizer",
                        type=str, 
                        default="bowphs/LaTa", 
                        help="Tokenizer to use for the search")
    parser.add_argument("--type",
                        type=str, 
                        default="roberta", 
                        help="Type of the model to use for the search")
    parser.add_argument("--index_model",
                        type=str, 
                        help="Type of the model to use for the search")
    args = parser.parse_args()

    # Commandline arguments
    FLAGS = flags.FLAGS
    flags.DEFINE_string("model", args.model, "Model to use for the search")
    flags.DEFINE_string("tokenizer", args.tokenizer, "Tokenizer to use for the search")
    flags.DEFINE_string("type", args.type, "Type of the model to use for the search")
    flags.DEFINE_string("index_model", args.index_model, "Type of the model to use for the search")
    config_flags.DEFINE_config_file("config", config_name, "configuration.", lock_config=True)
    flags.mark_flags_as_required(["config"])
    
    app.run(main)
