# nlp25_project_gr2
Repo for the exam project for NLP 2025


## Repo stucture
General:
- README.md
- .gitignore
- requirements.txt

eda.ipynb:

preprocess.ipynb: 

Data:
- da_news: contains the DaN+ data downloaded from ([GitHub](https://github.com/bplank/DaNplus)).
    - train (tsv)
    - dev (tsv)
    - test (tsv)

- me_data: contains the augmented versions of the non-overlapping dev and test sets with only MENAPT NERs.
    - dev (tsv and iob2)
    - test (tsv and iob2)

- me_entity_sources: the downloaded lists of MENAPT NERs.

- no_overlap_da_news: contains the non-overlapping splits of the DaN+ data.
    - train (tsv)
    - dev (tsv and iob2)
    - test (tsv and iob2)


evaluation:
- aug_preds: the predictions on the non-overlapping and the augmented dev sets for different versions of the train data.

- baseline_preds: the predictions on the non-overlapping and the augmented dev sets for the non-augmented train set.

- aug_eval.ipynb: notebook with evaluation of the experiments.

- baseline_eval.ipynb: notebook with evaluation of the baseline.

- output.txt: the output of using span_f1.py on the predictions.


hpc_jobs: TO DO (Når den er done)


scripts: all scripts used in the project. 
    - Note that span_f1.py was provided as part of the project template.