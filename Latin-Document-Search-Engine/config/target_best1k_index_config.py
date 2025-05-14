from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()


    config.run = run = ConfigDict()

    #gpu device
    run.device = 2

    config.data = data = ConfigDict()
    data.json_dataset_path = "/home/hawau/Latin-Document-Search-Engine/data/Latin_benchmark.txt"
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768
    data.window_data = 1

    config.model = model = ConfigDict()

    model.tokenizer = "bowphs/LaTa" # LaBerta,PhilBerta ,LaTa
    model.model = "bowphs/LaTa"
    model.type = "t5" # "roberta", "t5"


    model.model_max_length = 512 #max number of tokens the model can handle
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 50
    model.top_author = 1000
    model.filter_max_depth = 1000000

    config.index = index = ConfigDict()

    index.index_path = "/home/hawau/Latin-Document-Search-Engine/target_random1k_index"
    index.idx_2_keys = "/home/hawau/Latin-Document-Search-Engine/target_random1k_index/knn_{0}.json"
    index.index_name ="knn_target_{0}"

    # debug mode
    # index.index_path = "/home/hawau/Latin-Document-Search-Engine/debug_index"
    # index.idx_2_keys = "/home/hawau/Latin-Document-Search-Engine/debug_index/knn.json"

    config.db = db = ConfigDict()
    db.db_path = "/home/hawau/Latin-Document-Search-Engine"
    db.db_name ="DB_target_random1k_{0}"


    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 250

    return config
