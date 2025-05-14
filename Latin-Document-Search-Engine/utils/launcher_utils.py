import os
import json
import numpy as np
import torch
import faiss
from preprocessing_data.contractions import contractions, tokenize_clean

def get_best_results(index, H, cursor, query, tokenizer, model, k=1, device=None):

    encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)

    # Tokenize sentence
    inputs = tokenizer(query, return_tensors="pt").to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

    encoding[0] = sentence_embedding

    # Normalize
    faiss.normalize_L2(encoding)

    #search best matches in index
    distances, ann = index.search(encoding, k=k)


    json_result = {}

    r=0
    #retrieve best results from db
    for idx in ann[0]:
        cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
        rows = cursor.fetchall()
        json_result[r] = {}
        for row in rows:
            json_result[r]['author_id'] = row[1]
            json_result[r]['id'] = row[2]
            json_result[r]['name'] = row[3]
            json_result[r]['sentence'] = row[4]
            json_result[r]['citations'] = row[5]
            json_result[r]['book_name'] = row[6]
        r+=1
    
    return json_result


def custom_get_best_results(index, H, idx_2_keys, query, tokenizer, model, k=1, device=None, additional_text='', additional_text_slider_value=0.0):
    query = tokenize_clean(query)
    
    with torch.no_grad():
        tokenized = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**tokenized)
        model_cls = outputs.last_hidden_state[:, 0]
        model_cls = model_cls.cpu().detach().numpy()

    if additional_text != '':
        additional_text = tokenize_clean(additional_text)
        with torch.no_grad():
            tokenized_additional_text = tokenizer(additional_text, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs_additional_text = model(**tokenized_additional_text)
            additional_text_cls = outputs_additional_text.last_hidden_state[:, 0]
            additional_text_cls = additional_text_cls.cpu().detach().numpy()
            
        model_cls = model_cls*(1-additional_text_slider_value) + additional_text_cls*additional_text_slider_value

    faiss.normalize_L2(model_cls)
    
    #search best matches in index
    distances, ann = index.search(model_cls, k=k)
    
    json_result = {}
    r=0
    
    #retrieve best results
    for idx in ann[0]:
        idx_2_keys[idx]
        variable_split = idx_2_keys[idx].split('_')
        folder = '_'.join(variable_split[:-2])
        json_file = '_'.join(variable_split[:-1]) + '.json'
        
        # it is necessary to remove one element from the content_id because the index starts from 1 when we tokenize the batch
        content_id = int(variable_split[-1]) - 1
        
        file_open = os.path.join(H.data.json_dataset_path, folder, json_file)
        with open(file_open, 'r') as f:
            data_content = json.load(f)
        
        window_data = data_content['content'][content_id]
        if content_id - H.data.window_data > 0:
            window_data = data_content['content'][content_id-H.data.window_data] + ' ' + window_data
        if content_id + H.data.window_data < len(data_content['content']):
            window_data = window_data + ' ' + data_content['content'][content_id+H.data.window_data]
        
        json_result[r] = {}
        # json_result[r]['author_id'] = ...
        json_result[r]['id'] = idx_2_keys[idx]
        json_result[r]['name'] = data_content['author']
        json_result[r]['context'] = window_data
        json_result[r]['exact_match'] = data_content['content'][content_id]
        json_result[r]['book_name'] = data_content['title']
        
        r+=1
    
    return json_result

def custom_get_best_results_filtered(index, H, idx_2_keys, query, tokenizer, model, k=1, device=None, additional_text='', additional_text_slider_value=0.0, works_selected='All', max_depth=10):
    
    query = tokenize_clean(query)
    
    with torch.no_grad():
        tokenized = tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**tokenized)
        model_cls = outputs.last_hidden_state[:, 0]
        model_cls = model_cls.cpu().detach().numpy()

    if additional_text != '':
        additional_text = tokenize_clean(additional_text)
        with torch.no_grad():
            tokenized_additional_text = tokenizer(additional_text, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs_additional_text = model(**tokenized_additional_text)
            additional_text_cls = outputs_additional_text.last_hidden_state[:, 0]
            additional_text_cls = additional_text_cls.cpu().detach().numpy()
            
        model_cls = model_cls*(1-additional_text_slider_value) + additional_text_cls*additional_text_slider_value

    faiss.normalize_L2(model_cls)
    
    #search best matches in index on max_depth
    distances, ann = index.search(model_cls, k=max_depth)
    
    json_result = {}
    r=0
    finded = False
    
    #retrieve best results
    for val, idx in enumerate(ann[0]):
        idx_2_keys[idx]
        variable_split = idx_2_keys[idx].split('_')
        folder = '_'.join(variable_split[:-2])
        json_file = '_'.join(variable_split[:-1]) + '.json'
        
        if val == H.model.top_author:
            if finded == False:
                warning_author = True
            elif finded == True:
                warning_author = False
            else:
                raise ValueError('finded variable not set')
        if works_selected == 'All':
            warning_author = False
        
        # apply filter here 
        if works_selected == folder or works_selected == 'All':
            if val <= H.model.top_author:
                finded = True
            
            # it is necessary to remove one element from the content_id because the index starts from 1 when we tokenize the batch
            content_id = int(variable_split[-1]) - 1
            
            file_open = os.path.join(H.data.json_dataset_path, folder, json_file)
            with open(file_open, 'r') as f:
                data_content = json.load(f)
            
            window_data = data_content['content'][content_id]
            if content_id - H.data.window_data > 0:
                window_data = data_content['content'][content_id-H.data.window_data] + ' ' + window_data
            if content_id + H.data.window_data < len(data_content['content']):
                window_data = window_data + ' ' + data_content['content'][content_id+H.data.window_data]
            
            json_result[r] = {}
            json_result[r]['id'] = idx_2_keys[idx]
            json_result[r]['name'] = data_content['author']
            json_result[r]['context'] = window_data
            json_result[r]['exact_match'] = data_content['content'][content_id]
            json_result[r]['book_name'] = data_content['title']
            
            r+=1
                
        if r == k:
            break
    
    return json_result, warning_author
