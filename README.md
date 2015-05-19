Overview
========

This repository contains code for a shift-reduce discourse parser based on rhetorical structure theory.  A detailed system description can be found at http://arxiv.org/abs/1505.02425.


License
=======

This code is licensed under the MIT license (see LICENSE.txt).


Setup
=====

This code requires python 3.  I currently use 3.3.5.

This repository is pip-installable.  To make it work properly, I recommend running `pip install -e .` to set it up.  This will make a local, editable copy in your python environment.  See `requirements.txt` for a list of the prerequisite packages.  In addition, you may have to install a few NLTK models using `nltk.download()` in python (specifically, punkt and, at least for now, the maxent POS tagger).

Additionally, the syntactic parsing code must be set up to use ZPar.  The simplest but least efficient way is to put the ZPar distribution (version 0.6) in a subdirectory `zpar` (or symbolic link) in the current working directory, along with the English models in a subdirectory `zpar/english`.  For efficiency, a better method is to use the `python-zpar` wrapper, which is currently available at `https://github.com/EducationalTestingService/python-zpar` or `https://pypi.python.org/pypi/python-zpar/`.  To set this up, run make and then either a) set an environment variable `ZPAR_LIBRARY_DIR` equal to the directory where `zpar.so` is created (e.g., `/Users/USER1/python-zpar/dist`) to run ZPar as part of the discourse parser, or b) start a separate server using python-zpar's `zpar_server`.

Finally, CRF++ (version 0.58) should be installed, and its `bin` directory should be added to your `PATH` environment variable.  See `http://crfpp.googlecode.com/svn/trunk/doc/index.html`.

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

Visualization
=============

The script `util/visualize_rst_tree.py` can be used to create an HTML/javascript visualization, using D3.js (http://d3js.org/).  See the D3.js license: `util/LICENSE_d3.txt`.  The input to the script is the output of `rst_parse`.  See `util/example.json` for an example input.
