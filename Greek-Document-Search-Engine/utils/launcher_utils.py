
import numpy as np
import torch
import faiss

def get_best_results(index, H, cursor, query, tokenizer, model, k=1, max_documents=100, author_id=None, works_selected="All", additional_text='', additional_text_slider_value='', device=None):

    encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)

    # Tokenize sentence
    inputs = tokenizer(query, return_tensors="pt").to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

    if additional_text!='':

        # Tokenize sentence
        inputs = tokenizer(additional_text, return_tensors="pt").to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Use [CLS] token embedding as sentence encoding
        additional_sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32) 

        #alpha blending
        sentence_embedding = sentence_embedding*(1-additional_text_slider_value) + additional_sentence_embedding*additional_text_slider_value

    encoding[0] = sentence_embedding

    # Normalize
    faiss.normalize_L2(encoding)


    if author_id=='' and works_selected=="All":
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
    
    else:
        #search best matches in index
        distances, ann = index.search(encoding, k=max_documents)


        json_result = {}

        r=0
        #retrieve best results from db
        for idx in ann[0]:
            if r==k:
                break
            cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
            rows = cursor.fetchall()

            check_1 = True if author_id=='' else rows[0][1] == author_id
            check_2 = True if works_selected=='All' else rows[0][6] == works_selected

            if check_1 and check_2:
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