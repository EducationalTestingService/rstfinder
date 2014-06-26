
Setup
=====

This repository is pip-installable.  To make it work properly, I recommend running `pip install -e .` to set it up.  This will make a local, editable copy in your python environment.


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
tune_segmentation_model rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentationModel
```

Parsing
=======

To extract features for training a parsing model, run:

```
train_rst_parser rst_discourse_tb_edus_TRAINING_TRAIN.json > train_features
```

(TODO: After that, we need to figure out what to run to train a model, with, e.g., sklearn or SKLL.  Currently, the files have one action with features per line, with each line separated by features with the action in the first column followed by the active features.)



