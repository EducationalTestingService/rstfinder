
Setup
=====

This repository is pip-installable.  To make it work properly, I recommend running `pip install -e .` to set it up.  This will make a local, editable copy in your python environment.

Additionally, the code expects by default for a directory `zpar` to be in the current working directory.  This should contain the ZPar distribution (version 0.6), along with the English models in a subdirectory `english`.  This `zpar` directory can be a symbolic link.

Input Preparation
=================

To create a merged dataset that contains the RST Discourse Treebank along with the corresponding Penn Treebank parse trees for the same documents, run the following command (with paths adjusted as appropriate):

```
convert_rst_discourse_tb ~/corpora/rst_discourse_treebank ~/corpora/treebank_3
```

To split the RST discourse treebank training set into a new training and development set, run the following command:

```
discourseparsing/make_traindev_split.py
```

Segmentation
============

To create inputs (features and labels) for training a discourse segmentation model for the newly created training and development sets, run:

```
extract_segmentation_features rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv

extract_segmentation_features rst_discourse_tb_edus_TRAINING_DEV.json rst_discourse_tb_edus_features_TRAINING_DEV.tsv
```

To train (with the training set) and tune (with the development set) a discourse segmentation model, run:

```
tune_segmentation_model rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentation_model
```

Parsing
=======

To train an RST parsing model, run:

```
tune_rst_parser rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_TRAINING_DEV.json rst_parsing_model
```

To process a raw text document `my_document` with the end-to-end parser (assuming `C = 1.0` was the best hyperparameter setting according to `tune_segmentation_model`), run:

```
rst_parse -g segmentation_model.C1.0 -p rst_parsing_model.C1.0 my_document
```

Evaluation
==========

To evaluate an existing model, run:

```
rst_eval rst_discourse_tb_edus_TRAINING_DEV.json -p rst_parsing_modelC1.0 --use_gold_syntax
```

This will compute precision, recall, and F1 scores for 3 scenarios: spans labeled with nuclearity and relation types, spans labeled only with nuclearity, and unlabeled token spans.  The above version of the command will use gold standard EDUs and syntactic parses.

NOTE: The evaluation script has basic functionality in place, but at the moment it almost certainly does not appropriately handle important edge cases (e.g., same-unit relations, relations at the top of the tree).  These issues need to be addressed before the script can be used in experiments.
