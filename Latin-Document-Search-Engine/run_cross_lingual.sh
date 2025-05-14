models=(
    "bowphs/PhilBerta"
    "bowphs/SPhilBerta"
    "bowphs/PhilTa"
)
tokenizers=(
    "bowphs/PhilBerta"
    "bowphs/SPhilBerta"
    "bowphs/PhilTa"
)
types=(
    "roberta"
    "roberta"
    "t5"
)

m_length=${#models[@]}
language="Greek"
for ((j=0; j<m_length; j++)); do
    echo "Language: ${language}"
    echo "Running for ${models[j]} with ${tokenizers[j]} and ${types[j]}"
    python cross_lingual_search.py --config config/cross_lingual_config.py \
        --model "${models[j]}" \
        --tokenizer "${tokenizers[j]}" \
        --type "${types[j]}" \
        --query_lang "${language}"
done

language="Latin"
for ((j=0; j<m_length; j++)); do
    echo "Language: ${language}"
    echo "Running for ${models[j]} with ${tokenizers[j]} and ${types[j]}"
    python cross_lingual_search.py --config config/cross_lingual_config.py \
        --model "${models[j]}" \
        --tokenizer "${tokenizers[j]}" \
        --type "${types[j]}" \
        --query_lang "${language}"
done




# python assess.py --index_model "LaBerta" 
# python assess.py --index_model "PhilBerta"
# python assess.py --index_model "LaTa"