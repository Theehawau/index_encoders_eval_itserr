import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback


# MODELLO BASE
model_name = "bowphs/SPhilBerta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# Funzione per caricare le frasi
def load_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# MODELLO SIMCSE WRAPPER
class SimCSEModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]  # CLS token

model = SimCSEModel(base_model)

# CARICAMENTO CSV: modifica qui il path al tuo file
dataset = load_dataset("csv", data_files={"train": "./Dataset_SimCSE_all_in.csv"})["train"]

# TOKENIZZAZIONE DOPPIA
def tokenize(example):
    tok1 = tokenizer(example["anchor"], padding="max_length", truncation=True, max_length=128)
    tok2 = tokenizer(example["positive"], padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids": tok1["input_ids"],
        "attention_mask": tok1["attention_mask"],
        "input_ids_b": tok2["input_ids"],
        "attention_mask_b": tok2["attention_mask"],
    }

dataset = dataset.train_test_split(test_size=0.1, seed=42)  # Split del dataset in train e test
dataset = dataset.map(tokenize)

# DATASET TORCH
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "input_ids_b": torch.tensor(row["input_ids_b"]),
            "attention_mask_b": torch.tensor(row["attention_mask_b"]),
        }

# CONTRASTIVE LOSS
class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, emb1, emb2):
        sim = self.cos(emb1.unsqueeze(1), emb2.unsqueeze(0)) / self.temperature
        labels = torch.arange(emb1.size(0)).to(emb1.device)
        return nn.CrossEntropyLoss()(sim, labels)

def get_embeddings(path, model, tokenizer):
    sentences = load_sentences(path)
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.inference_mode():
        embeddings = model(**tokenized_inputs.to(model.encoder.device)).cpu()
    return embeddings

def compute_similarity(embeddings1, embeddings2):
    num = (embeddings1 @ embeddings2.T)
    denom = (
        embeddings1.norm(dim=-1, keepdim=True) *
        embeddings2.norm(dim=-1, keepdim=True).T
    )
    similarity = num / denom
    return similarity

def compute_recall(similarity_matrix, out_dict, k=10, n=100):
    # Ugly oneliner
    n_matches = (similarity_matrix.argsort(dim=-1)[:, -k:].fmod(n) == torch.arange(n).unsqueeze(1)).sum(dim=-1)
    recall_k = (n_matches / 5).mean()
    return recall_k.item(), n_matches.sum().item()


# TRAINER PERSONALIZZATO
class SimCSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_input = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        positive_input = {
            "input_ids": inputs["input_ids_b"],
            "attention_mask": inputs["attention_mask_b"],
        }

        anchor_output = model(**anchor_input)
        positive_output = model(**positive_input)

        loss = SimCSELoss()(anchor_output, positive_output)
        return (loss, (anchor_output, positive_output)) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model.eval()

        # Inizializza le variabili per la valutazione
        total_loss = 0.0
        num_samples = 0

        for _, inputs in enumerate(eval_dataloader):
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)

            total_loss += loss.item() * inputs["input_ids"].size(0)
            num_samples += inputs["input_ids"].size(0)

        avg_loss = total_loss / num_samples
        out_dict = {"eval_loss": avg_loss}

        # greek_queries = load_sentences("./100_query_greek.txt")
        # greek_targets = load_sentences("./500_target_greek.txt")
        # latin_queries = load_sentences("./100_query_latin.txt")
        # latin_targets = load_sentences("./500_target_latin.txt")

        # tokenized_greek_queries = tokenizer(greek_queries, padding=True, truncation=True, return_tensors="pt", max_length=128)
        # tokenized_greek_targets = tokenizer(greek_targets, padding=True, truncation=True, return_tensors="pt", max_length=128)
        # tokenized_latin_queries = tokenizer(latin_queries, padding=True, truncation=True, return_tensors="pt", max_length=128)
        # tokenized_latin_targets = tokenizer(latin_targets, padding=True, truncation=True, return_tensors="pt", max_length=128)

        # with torch.inference_mode():
        #     greek_queries_embeddings = model(**tokenized_greek_queries.to(model.encoder.device)).cpu()
        #     greek_targets_embeddings = model(**tokenized_greek_targets.to(model.encoder.device)).cpu()
        #     latin_queries_embeddings = model(**tokenized_latin_queries.to(model.encoder.device)).cpu()
        #     latin_targets_embeddings = model(**tokenized_latin_targets.to(model.encoder.device)).cpu()
        greek_syn_q_embs = get_embeddings("100_query_greek.txt", model, tokenizer)
        greek_syn_t_embs = get_embeddings("500_target_greek.txt", model, tokenizer)
        latin_syn_q_embs = get_embeddings("100_query_latin.txt", model, tokenizer)
        latin_syn_t_embs = get_embeddings("500_target_latin.txt", model, tokenizer)
        greek_real_q_embs = get_embeddings("greco_latino_veri/20_query_greek.txt", model, tokenizer)
        latin_real_q_embs = get_embeddings("greco_latino_veri/20_query_latin.txt", model, tokenizer)
        greek_real_t_embs = get_embeddings("greco_latino_veri/100_target_greek.txt", model, tokenizer)
        latin_real_t_embs = get_embeddings("greco_latino_veri/100_target_latin.txt", model, tokenizer)

        greek_syn_similarity = compute_similarity(greek_syn_q_embs, greek_syn_t_embs)
        greek_real_similarity = compute_similarity(greek_real_q_embs, greek_real_t_embs)
        latin_syn_similarity = compute_similarity(latin_syn_q_embs, latin_syn_t_embs)
        latin_real_similarity = compute_similarity(latin_real_q_embs, latin_real_t_embs)
        greek_latin_real_similarity = compute_similarity(greek_real_q_embs, latin_real_t_embs)
        latin_greek_real_similarity = compute_similarity(latin_real_q_embs, greek_real_t_embs)

        for k in [5, 10]:
            greek_syn_recall_k, greek_syn_n_matches = compute_recall(greek_syn_similarity, out_dict, k)
            latin_syn_recall_k, latin_syn_n_matches = compute_recall(latin_syn_similarity, out_dict, k)
            out_dict[f"greek syn Rec@{k}"] = greek_syn_recall_k
            out_dict[f"greek syn N_matches@{k}"] = greek_syn_n_matches
            out_dict[f"latin syn Rec@{k}"] = latin_syn_recall_k
            out_dict[f"latin syn N_matches@{k}"] = latin_syn_n_matches

            greek_real_recall_k, greek_real_n_matches = compute_recall(greek_real_similarity, out_dict, k, n=20)
            latin_real_recall_k, latin_real_n_matches = compute_recall(latin_real_similarity, out_dict, k, n=20)
            out_dict[f"greek real Rec@{k}"] = greek_real_recall_k
            out_dict[f"greek real N_matches@{k}"] = greek_real_n_matches
            out_dict[f"latin real Rec@{k}"] = latin_real_recall_k
            out_dict[f"latin real N_matches@{k}"] = latin_real_n_matches

            greek_latin_real_recall_k, greek_latin_real_n_matches = compute_recall(greek_latin_real_similarity, out_dict, k, n=20)
            latin_greek_real_recall_k, latin_greek_real_n_matches = compute_recall(latin_greek_real_similarity, out_dict, k, n=20)
            out_dict[f"greek-latin real Rec@{k}"] = greek_latin_real_recall_k
            out_dict[f"greek-latin real N_matches@{k}"] = greek_latin_real_n_matches
            out_dict[f"latin-greek real Rec@{k}"] = latin_greek_real_recall_k
            out_dict[f"latin-greek real N_matches@{k}"] = latin_greek_real_n_matches

        self.log(out_dict)
        
        # Multilingual Eval
        
        return out_dict

# ARGOMENTI DI TRAINING
training_args = TrainingArguments(
    output_dir="./simcse_sphilberta_output",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    fp16=True,  # Mixed precision
    learning_rate=5e-5,
    
    logging_strategy="steps",
    logging_steps=20,
    
    eval_strategy="steps",
    eval_steps=20,
    
    save_strategy="steps",
    save_steps=20,
    
    remove_unused_columns=False,
    max_grad_norm=1.0,
    overwrite_output_dir=True,


    # flag early stopping
    load_best_model_at_end=True,        # salva e restore del best checkpoint
    metric_for_best_model="eval_loss",  # metrica su cui basarsi
    greater_is_better=False,            # per la loss, più basso è meglio
    save_total_limit=3,                 # conserva solo gli ultimi 3 checkpoint

)

# TRAINER
trainer = SimCSETrainer(
    model=model,
    args=training_args,
    train_dataset=ContrastiveDataset(dataset["train"]),
    eval_dataset=ContrastiveDataset(dataset["test"]),  # Usato per la validazione
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # stop se non migliora per 2 valutazioni

)

out_dict = trainer.evaluate()  # Valutazione iniziale
print(out_dict)

# LANCIO DEL TRAINING
trainer.train()


# SALVATAGGIO DEL MODELLO E DEL TOKENIZER
model_save_path = "./simcse_sphilberta_model"
tokenizer_save_path = "./simcse_sphilberta_tokenizer"

# Salva il modello
model.encoder.save_pretrained(model_save_path)

# Salva il tokenizer
tokenizer.save_pretrained(tokenizer_save_path)

print("Modello e tokenizer salvati.")

