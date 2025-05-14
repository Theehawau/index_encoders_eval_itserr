# python my_search.py --config config/index_config.py \
#     --get_targets --get_similarity \
#     --model "bowphs/LaBerta" \
#     --tokenizer "bowphs/LaBerta" \
#     --type "roberta" 

models=(
    # "bowphs/LaBerta"
    "bowphs/PhilBerta"
    "bowphs/LaTa"
)
tokenizers=(
    # "bowphs/LaBerta"
    "bowphs/PhilBerta"
    "bowphs/LaTa"
)
types=(
    # "roberta"
    "roberta"
    "t5"
)

# m_length=${#models[@]}
# for ((j=0; j<m_length; j++)); do
#     echo "Running for ${models[j]} with ${tokenizers[j]} and ${types[j]}"
#     python my_search.py --config config/index_config.py \
#         --get_similarity \
#         --model "${models[j]}" \
#         --tokenizer "${tokenizers[j]}" \
#         --type "${types[j]}" 
# done



# --config config/target_best1k_index_config.py
# python create_target_random1k_index.py  \
#     --model "bowphs/LaBerta" \
#     --tokenizer "bowphs/LaBerta" \
#     --type "roberta" \
#     --index_model "LaBerta"

# python create_target_random1k_index.py  \
#     --model "bowphs/PhilBerta" \
#     --tokenizer "bowphs/PhilBerta" \
#     --type "roberta" \
#     --index_model "PhilBerta"

# python create_target_random1k_index.py  \
#     --model "bowphs/LaTa" \
#     --tokenizer "bowphs/LaTa" \
#     --type "t5" \
#     --index_model "LaTa"

# # python my_search.py --config /home/hawau/Latin-Document-Search-Engine/config/index_config.py --get_targets 
# # --get_similarity 


# echo "Running for LaBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_targets --get_similarity \
#     --model "bowphs/LaBerta" \
#     --tokenizer "bowphs/LaBerta" \
#     --type "roberta" \
#     --index_model "LaBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/PhilBerta" \
#     --tokenizer "bowphs/PhilBerta" \
#     --type "roberta" \
#     --index_model "LaBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/LaTa" \
#     --tokenizer "bowphs/LaTa" \
#     --type "t5" \
#     --index_model "LaBerta"


# echo "Running for PhilBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_targets --get_similarity \
#     --model "bowphs/PhilBerta" \
#     --tokenizer "bowphs/PhilBerta" \
#     --type "roberta" \
#     --index_model "PhilBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/LaBerta" \
#     --tokenizer "bowphs/LaBerta" \
#     --type "roberta" \
#     --index_model "PhilBerta"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/LaTa" \
#     --tokenizer "bowphs/LaTa" \
#     --type "t5" \
#     --index_model "PhilBerta"


# echo "Running for LaTa"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_targets --get_similarity \
#     --model "bowphs/LaTa" \
#     --tokenizer "bowphs/LaTa" \
#     --type "t5" \
#     --index_model "LaTa"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/PhilBerta" \
#     --tokenizer "bowphs/PhilBerta" \
#     --type "roberta" \
#     --index_model "LaTa"

# python search_my_target_best1k_index.py --config /home/hawau/Latin-Document-Search-Engine/config/target_best1k_index_config.py \
#     --get_similarity \
#     --model "bowphs/LaBerta" \
#     --tokenizer "bowphs/LaBerta" \
#     --type "roberta" \
#     --index_model "LaTa"


# python assess.py --index_model "LaBerta" 
# python assess.py --index_model "PhilBerta"
# python assess.py --index_model "LaTa"