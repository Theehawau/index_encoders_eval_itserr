from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()


    config.run = run = ConfigDict()

    #gpu device
    run.device = 2

    config.data = data = ConfigDict()

    '''
        The dataset is in this format:
            - dataset_name
                - author_1
                    - <...>.json
                    - <...>.json
                    ...
                    - <...>.json
                - author_2
                    - <...>.json
                    - <...>.json
                    ...
                    - <...>.json
                ...
        
        So, json_dataset_path is the absolute path to 'dataset_name'
    '''
    data.json_dataset_path = "/home/hawau/tlg"

    '''
    When creating the index, sentences are broken up into low and high points. However, some sentences, if broken up, do not make much sense on their own.

    e.g. ...τῆς βασιλείας αὐτοῦ νῦν καὶ εἰς τοὺς αἰῶνας τῶν αἰώνων. ἀμήν.

    It is a prayer that ends with "amen". It would be ideal to keep it in the sentence.
    The min_words_in_phrase parameter indicates how many words at least a sentence must have to be "alone", separate from the others.
    '''
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768

    #dove si trova la lista contenente le opere TLG (create prima con lo script save_name_of_the_works)
    data.name_of_the_works_path = "/home/hawau/name_of_the_tlg_works.npy"


    config.model = model = ConfigDict()

    # model.tokenizer = 'bowphs/GreTa' #"bowphs/PhilTa"
    # model.model = 'bowphs/GreTa'
    # model.type = "t5"

    model.tokenizer = '/home/hawau/SPhilBERTa_MLM_final2' # 'bowphs/GreBerta', 'bowphs/PhilBerta', "/home/hawau/PhilBERTa_MLM_final2"
    model.model = '/home/hawau/SPhilBERTa_MLM_final2'
    model.type = "roberta" # "roberta", "sentence-transformers", "bert", "t5"

    #max number of tokens the model can handle
    model.model_max_length = 512
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 10


    config.index = index = ConfigDict()
    index.index_path = "/home/hawau/"
    index.index_name ="Faiss_Greek"

    config.db = db = ConfigDict()
    db.db_path = "/home/hawau/"
    db.db_name ="DB_Greek"



    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 1000

    return config