from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import json
# from utils.launcher_utils import get_best_results
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3
import numpy as np
import ast
import unicodedata
import re
from tqdm import tqdm
import argparse



def normalize_text(text):
    # Rimuovere caratteri invisibili come spazi non separabili
    text = text.replace('\xa0', ' ')  # Sostituire \xa0 con uno spazio
    # Normalizzazione Unicode
    text = unicodedata.normalize('NFKC', text)
    # Rimuovere eventuali spazi extra all'inizio e alla fine
    text = text.strip()
    # Rimuovere caratteri invisibili come nuove righe e tabulazioni
    text = re.sub(r'\s+', ' ', text)  # Sostituire sequenze di spazi con un singolo spazio
    return text


def compute_similarity_faiss(a, b, H):
    index = faiss.IndexFlatL2(H.data.len_embedding)
    # Add one embedding to the index
    index.add(np.array([a]))  # Add as base

    # Search for top-1 most similar in the index to embedding_2
    D, I = index.search(np.array([b]), k=1)
    return D[0][0]

def compute_cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    similarity_score = np.dot(a, b)
    return similarity_score


def main(argv):

    H = FLAGS.config
    H.model.tokenizer = FLAGS.tokenizer
    H.model.model = FLAGS.model
    H.model.type = FLAGS.type

    H.index.index_name = H.index.index_name.format(FLAGS.index_model)
    H.db.db_name = H.db.db_name.format(FLAGS.index_model)


    device = H.run.device if torch.cuda.is_available() else -1


    #load tokenizer and model
    if H.model.type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
        model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)
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

    
    out_dir = f"{H.data.save_index_output_dir}/{FLAGS.index_model}_index/_outputs_{H.model.model.replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)
    print("Saving outputs to:", out_dir)


    #load index
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}.index"))
    path_to_load_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_load_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot open database."

    cursor = connection.cursor()

    df = pd.read_csv(f"{H.data.test_benchmark_path}", sep="\t")

    columns = df.columns.to_list()
    columns.extend([ f"best #{i}" for i in range(H.retrieval.num_matches)])

    print("Get targets:", FLAGS.get_targets)
    print("Get similarity:", FLAGS.get_similarity)

    if FLAGS.get_targets:
        out_file = open(f"{H.data.save_index_output_dir}/{FLAGS.index_model}_index/Greek_benchmark_best_targets.txt", "w", encoding="utf-8")
        print('\t'.join(columns), file=out_file)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Queries", unit="query"):
            text = row['Query']
            print('\t'.join(row.to_list()), file=out_file, end="\t")
            #search best matches in index
            distances, ann = get_distance_target(text, tokenizer, device, model, index, H, cursor)
            outs = []
            #retrieve best results from db
            for i,idx in enumerate(ann):
                cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
                rows = cursor.fetchall()
                for row in rows:
                    outs.append(f"{row[1]}")

            print('\t'.join(outs), file=out_file)

        out_file.close()

    df = pd.read_csv(f"{H.data.save_index_output_dir}/{FLAGS.index_model}_index/Greek_benchmark_best_targets.txt", sep="\t")

    if FLAGS.get_similarity:
        out_file = open(f"{out_dir}/Greek_benchmark_best_faiss_distances.txt", "w", encoding="utf-8")
        out_file_2 = open(f"{out_dir}/Greek_benchmark_best_cosine_similarities.txt", "w", encoding="utf-8")
        print('\t'.join(columns), file=out_file)
        print('\t'.join(columns), file=out_file_2)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Similarity", unit="query"):
            text = row['Query']
            similarities = [text]
            cosine_similarities = [text]
            a = np.zeros((1, H.data.len_embedding)).astype(np.float32)
            if H.model.type != "sentence-transformers":
                # Tokenize sentence
                inputs = tokenizer(text, return_tensors="pt").to(device)
                
                if len(inputs['input_ids'][0]) > H.model.model_max_length:
                    print(f"Query is too long: {text}")
                    continue

                # Get embeddings
                with torch.no_grad():
                    if H.model.type == "t5":
                        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                    else:
                        outputs = model(**inputs, output_hidden_states=True)

                # Use [CLS] token embedding as sentence encoding
                a[0] = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

            else:
                with torch.no_grad():
                    a[0] = model.encode(text, convert_to_tensor=True).squeeze().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(a)

            for target in row[1:]:
                b = np.zeros((1, H.data.len_embedding)).astype(np.float32)
                if H.model.type != "sentence-transformers":
                    # Tokenize sentence
                    inputs = tokenizer(normalize_text(str(target)), return_tensors="pt", truncation=True, max_length=512).to(device)
               
                # Get embeddings
                with torch.no_grad():
                    try:
                        if H.model.type == "t5":
                            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                        elif H.model.type == "sentence-transformers":
                            b[0] = model.encode(normalize_text(str(target)), convert_to_tensor=True).squeeze().cpu().numpy().astype(np.float32)
                        else:
                            outputs = model(**inputs, output_hidden_states=True)
                    except Exception as e:
                        print(f"Error processing target {target}")
                        continue
                if H.model.type != "sentence-transformers":
                    # Use [CLS] token embedding as sentence encoding
                    b[0] = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)
                faiss.normalize_L2(b)

                sim = compute_similarity_faiss(a[0], b[0], H)
                cosine_sim = compute_cosine_similarity(a[0], b[0])
                similarities.append(f"{sim:.4f}")
                cosine_similarities.append(f"{cosine_sim:.4f}")
                

            print('\t'.join(similarities), file=out_file)
            print('\t'.join(cosine_similarities), file=out_file_2)


        out_file.close()
        out_file_2.close()



def get_distance_target(text, tokenizer, device, model, index, H, cursor):
    encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)

    # Tokenize sentence
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get embeddings
    with torch.no_grad():
        # outputs = model(**inputs, output_hidden_states=True)
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        # sentence_embedding = model.encode(text, convert_to_tensor=True).squeeze().cpu().numpy().astype(np.float32)



    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

    encoding[0] = sentence_embedding

    # Normalize
    faiss.normalize_L2(encoding)

    #search best matches in index
    distances, ann = index.search(encoding, k=H.retrieval.num_matches)

    return distances[0], ann[0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        default="/home/hawau/Greek-Document-Search-Engine/config/index_config.py", 
                        help='Path to the config file')
    parser.add_argument("--get_targets", 
                        action='store_true', 
                        help="Get targets from the database")
    parser.add_argument("--get_similarity",
                        action='store_true', 
                        help="Get targets from the database")
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
    config_file = args.config
    get_targets = args.get_targets
    get_similarity = args.get_similarity


    FLAGS = flags.FLAGS

    flags.DEFINE_bool("get_targets", get_targets, "Get targets from the database")
    flags.DEFINE_bool("get_similarity", get_similarity, "Get targets from the database")
    flags.DEFINE_string("model", args.model, "Model to use for the search")
    flags.DEFINE_string("tokenizer", args.tokenizer, "Tokenizer to use for the search")
    flags.DEFINE_string("type", args.type, "Type of the model to use for the search")
    flags.DEFINE_string("index_model", args.index_model, "Type of the model to use for the search")
    config_flags.DEFINE_config_file("config", config_file, "configuration.", lock_config=True)
    flags.mark_flags_as_required(["config"], )
    app.run(main)