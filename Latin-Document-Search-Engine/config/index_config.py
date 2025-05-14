from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()


    config.run = run = ConfigDict()

    #gpu device
    run.device = 1

    config.data = data = ConfigDict()
    data.json_dataset_path = "/home/hawau/Latin-Document-Search-Engine/db_data"
    data.test_benchmark_path = "/home/hawau/Latin-Document-Search-Engine/data_parallel/Latin_Benchmark.txt"
    data.save_index_output_dir = "data_parallel"
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768
    data.window_data = 1

    config.model = model = ConfigDict()

    model.tokenizer = "bowphs/SPhilBerta"
    model.model = "bowphs/SPhilBerta"
    model.type = "sentence-transformers"


    model.model_max_length = 512 #max number of tokens the model can handle
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 30
    model.top_author = 1000
    model.filter_max_depth = 1000000

    config.index = index = ConfigDict()

    index.index_path = "/home/hawau/Latin-Document-Search-Engine/share_index"
    index.idx_2_keys = "/home/hawau/Latin-Document-Search-Engine/share_index/knn.json"
    index.index_name ="knn"

    # debug mode
    # index.index_path = "/home/hawau/Latin-Document-Search-Engine/debug_index"
    # index.idx_2_keys = "/home/hawau/Latin-Document-Search-Engine/debug_index/knn.json"

    config.db = db = ConfigDict()
    db.db_path = "/home/hawau/Latin-Document-Search-Engine"
    db.db_name ="DB_Latin"


    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 1000

    return config
