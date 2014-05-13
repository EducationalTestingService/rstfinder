discourse-parsing
=================

To create a merged dataset that contains the RST Discourse Treebank along with
the corresponding Penn Treebank parse trees for the same documents, run the
following command (with paths adjusted as appropriate):

```
./convert_rst_discourse_tb.py ~/corpora/rst_discourse_treebank ~/corpora/treebank_3
```

To split the RST discourse treebank training set into a new training and
development set, run the following command:

```
./make_traindev_split.py
```

To create inputs (features and labels) for training a discourse segmentation
model for the newly created training and development sets, run:

```
./extract_segmentation_features.py rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv

./extract_segmentation_features.py rst_discourse_tb_edus_TRAINING_DEV.json rst_discourse_tb_edus_features_TRAINING_DEV.tsv
```

To train (with the training set) and tune (with the development set) a
discourse segmentation model, run:

```
./tune_segmentation_model.py rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentationModel
```
