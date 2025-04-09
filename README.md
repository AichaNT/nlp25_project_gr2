# nlp25_project_gr2
Repo for the exam project for NLP 2025


## Repo stucture
General:
- README.md
- .gitignore
- requirements.txt


Data:
- DaN+ (danish news data)
    - train
    - dev
    - test

- MENAPT_NER:
    - B-PER
    - I-PER
    - ORG (both B/I)
    - LOC (both B/I)
    (- MISC??)


img:
- all images


Scripts: (ryk rundt alt efter hvad der bedst passer hvor)
- load_data.py
    - tsv2conll()
    - 

- data_quality_check.py
    - no_overlap()
    - 

- data_augmentation.py
    - augment_dataset() (used for MENAPT_test, MENAPT_train and MENAPT_dev)
        - ^^should maybe be dif functions based on what we need

- self_train.py
    - webscrape()
    - self_train()


- eval.py
    - span_f1()
    - significance_test()
    - metrics() (whatever leftover metrics we want)

- eda.py
    - tag_dist()
    - 