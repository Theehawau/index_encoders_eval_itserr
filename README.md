# Gretino: A Retrieval benchmark in Latin and Ancient Greek

The dataset is available in `data/` one can use pandas to read it:

```python
import pandas as pd

greek_silver = pd.read_csv("data/Greek_benchmark.txt", sep="\t")
greek_gold = pd.read_csv("data/Greek_benchmark_parallel.txt", sep="\t")
latin_silver = pd.read_csv("data/Latin_benchmark.txt", sep="\t")
latin_gold = pd.read_csv("data/Latib_benchmark_parallel.txt", sep="\t")
```

Each csv has a `Query` column and 5 target coluns `[f"Target #{i}" for i in range(1,6)]`.

## Model Evaluation

All the evaluation can be run in a notebook, `data_study.ipynb` and to replicate the tables in the paper with the nice formattting and colors one can run the `plots.ipynb` notebook after running the `data_study.ipynb`

## Fine Tuning

To fine-tune the SPhilBERTa model you can run the 2 following commands, one for each training dataset:

```bash
python fine_tune_simcse.py --data_path training_data/SimCSE25.csv
python fine_tune_simcse.py --data_path training_data/SimCSE21.csv
```
