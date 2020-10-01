![Travis CI Badge](https://img.shields.io/travis/EducationalTestingService/rstfinder) ![Conda Package](https://img.shields.io/conda/v/ets/rstfinder.svg) ![Conda Platform](https://img.shields.io/conda/pn/ets/rstfinder.svg) ![License](https://img.shields.io/github/license/EducationalTestingService/rstfinder)

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Usage](#usage)
   * [Train models](#train-models)
   * [Use trained models](#use-trained-models)
* [License](#license)


## Introduction

This repository contains the code for **RSTFinder** -- a discourse segmenter & shift-reduce parser based on rhetorical structure theory.  A detailed system description can be found in this [paper](http://arxiv.org/abs/1505.02425).

## Installation

RSTFinder currently works only on Linux and requires Python 3.6, 3.7, or 3.8. 

The only way to install RSTFinder is by using the `conda` package manager. If you have already installed `conda`, you can skip straight to Step 2.

1. To install `conda`, follow the instructions on [this page](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

2. Create a new conda environment (say, `rstenv`) and install the RSTFinder conda package in it.

    ```bash
    conda create -n rstenv -c conda-forge -c ets python=3.8 rstfinder
    ```

3. Activate this conda environment by running `conda activate rstfinder`. 

4. Now install the `python-zpar` package via `pip` in this environment. This package allows us to use the ZPar constituency parser (more later).

    ```bash
    pip install python-zpar
    ```

5. From now on, you will need to activate this conda environment whenever you want to use RSTFinder. This will ensure that the packages required by RSTFinder will not affect other projects.

## Usage

RSTFinder is trained using [RST Discourse Treebank](https://catalog.ldc.upenn.edu/LDC2002T07) and the [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42). However, these treebanks are not freely available and can only be accessed via a personal/academic/institutional subscription to the Linguistic Data Consortium (LDC). This means that we cannot make the RSTFinder parser models publicly available. However, we provide detailed instructions for users so that they can train their own RSTFinder models once they do have access to the treebanks.

### Train models

1. **Activate the conda environment.** Activate the previously created `rstenv` conda environment (see [installation](#installation)):

    ```bash
    conda activate rstenv
    ```

2. **Download NLTK tagger model**. Due to a rare mismatch between the RST Discourse Treebank and the Penn Treebank documents, sometimes there are parts of the document for which we cannot locate the corresponding parse trees. To get around this issue, we first sentence-tokenize & part-of-speech tag such parts using the MaxEnt POS tagger model from NLTK and, then, just create fake, shallow trees for them. Therefore, we need to download tokenizer and tagger models for this.

    ```bash
    export NLTK_DATA="$HOME/nltk_data"
    python -m nltk.downloader maxent_treebank_pos_tagger punkt
    ```

2. **Pre-process and merge the treebanks**. To create a merged dataset that contains the RST Discourse Treebank along with the corresponding Penn Treebank parse trees for the same documents, run the following command (with paths adjusted as appropriate):

    ```bash
    convert_rst_discourse_tb ~/corpora/rst_discourse_treebank ~/corpora/treebank_3
    ```

    where ` ~/corpora/rst_discourse_treebank` is the directory that contains the RST Discourse Treebank files. If you obtained this treebank from the LDC, then this is the directory that contains the `index.html` file. Similarly, `~/corpora/treebank_3` is the directory that contains the Penn Treebank files. If you obtained this treebank from the LDC, then this is the directory that contains the `parsed` sub-directory.

3. **Create a development set**. Split the documents in the RST discourse treebank training set into a new training and development set:

    ```bash
    make_traindev_split.py
    ```

    At the end of this command, you will have the following JSON files in your current directory:

    - `rst_discourse_tb_edus_TRAINING.json` : the original RST Discourse Treebank training set merged with the corresponding Penn Treebank trees in JSON format. 

    - `rst_discourse_tb_edus_TEST.json` : the original RST Discourse Treebank test set merged with the corresponding Penn Treebank trees in JSON format. 

    - `rst_discourse_tb_edus_TRAINING_DEV.json`: the development set split from `rst_discourse_tb_edus_TRAINING.json`. This fill will be used to tune the segmenter and RST parser hyperparameters. 

    - `rst_discourse_tb_edus_TRAINING_TRAIN.json` : the training set split from `rst_discourse_tb_edus_TRAINING.json`. This file will be used to train the segmenter and the parser. 

4. **Extract the segmenter features**. Create inputs (features and labels) to train a discourse segmentation model from the newly created training set:

    ```bash
    extract_segmentation_features rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv
    ```

    and the development set:

    ```bash
    extract_segmentation_features rst_discourse_tb_edus_TRAINING_DEV.json rst_discourse_tb_edus_features_TRAINING_DEV.tsv
    ```

    The extracted features for the training and development set are now in the `rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv` and `rst_discourse_tb_edus_features_TRAINING_DEV.tsv` files respectively.

5. **Train the CRF segmenter model and tune its hyper-parameters**. Train (with the training set) and tune (with the development set) a CRF-based discourse segmentation model:

    ```bash
    tune_segmentation_model rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentation_model
    ```

    This command iterates over a pre-defined list of values for the `C` regularization parameter for the CRF, trains a model using the features extracted from the training set, and then evaluates that model on the development set. Its final output is the `C` value that yields the highest performance F1 score on the development set. After this command, you will have a number of files with the prefix `segmentation_model` in the current directory, e.g., `segmentation_model.C0.25`, `segmentation_model.C1.0` et cetera. These are the CRF model files trained with those specific values of the `C` regularization parameter. Underlyingly, the command uses the `crf_learn` and `crf_test` binaries from [CRFPP](https://github.com/taku910/crfpp) via `subprocess`. 

6. **Train the logistic regression RST Parsing model and tune its hyper-parameters**. Train (with the training set) and tune (with the development set) a discourse parsing model that uses logistic regression:

    ```bash
    tune_rst_parser rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_TRAINING_DEV.json rst_parsing_model
    ```

    This command iterates over a pre-defined list of values for the `C` regularization parameter for logistic regression, trains a model using the features extracted from the training set, and then evaluates that model on the development set. Its final output is the `C` value that yields the highest performance F1 score on the development set. After this command, you will have a number of directories with the prefix `rst_parsing_model` in the current directory, e.g., `rst_parsing_model.C0.25`, `segmentation_model.C1.0` et cetera. Each of these directories contains the logistic regression model files (named `rst_parsing_all_feats_LogisticRegression.model`) trained with those specific values of the `C` regularization parameter.  Underlyingly, this command uses the [SKLL](https://skll.readthedocs.io) machine learning library to train and evaluate the models.

7. (Optional) **Evaluate trained model**. If you want to obtain detailed evaluation metrics for an RST parsing model on the development set, run:

    ```bash
    rst_eval rst_discourse_tb_edus_TRAINING_DEV.json -p rst_parsing_model.C1.0 --use_gold_syntax
    ```

    Of course, you could also use the test set here (`rst_discourse_tb_edus_TEST.json`) if you wished to do so.

    This command will compute precision, recall, and F1 scores for 3 scenarios: spans labeled with nuclearity and relation types, spans labeled only with nuclearity, and unlabeled token spans.  `--use_gold_syntax` means that the command will use gold standard EDUs and syntactic parses.

    *NOTE*: While the evaluation script has basic functionality in place, at the moment it almost certainly does not appropriately handle important edge cases (e.g., same-unit relations, relations at the top of the tree). 

### Use trained models

At this point, we are ready to use the segmentation and RST parsing models to process raw text documents. Before we do that, you will need to download some models for the [ZPar parser](https://github.com/frcchang/zpar).  RSTFinder uses ZPar to generate constituency parses for new documents. These models can be downloaded from [here](https://github.com/frcchang/zpar/releases/download/v0.7.5/english-models.zip). Uncompress the models into a directory of your choice, say `$HOME/zpar-models`. 

Next, you need to set the following environment variables:

```bash
export NLTK_DATA="$HOME/nltk_data"
export ZPAR_MODEL_DIR="$HOME/zpar-models"
```

Now we are good to go! To process a raw text document `document.txt` with the end-to-end parser (assuming `C` = 1.0 was the best hyper-parameter value for both the segmentation and RST parsing models), run:

```bash
rst_parse -g segmentation_model.C1.0 -p rst_parsing_model.C1.0 document.txt > output.json
```

`output.json` contains a dictionary with two keys: `edu_tokens` and `scored_rst_trees`. The value corresponding to `edu_tokens` is a list of lists;  each constituent list contains the tokens in an Elementary Discourse Unit (EDU) as computed by the segmenter. The value corresponding to `rst_trees` is a list of dictionaries: each dictionary has two keys, `tree` and `score` containing the RST parse tree for the document and its score respectively. By default, only a single tree is produed but additonal trees can be produced by specifying the `-n` option for `rst_parse`. 

RSTFinder can also produce an HTML/Javascript visualization of the RST parse tree using [D3.js](http://d3js.org/). To produce such a visualization from the JSON output file, run:

```bash
visualize_rst_tree output.json tree.html --embed_d3js
```

This will produce a self-contained file called `tree.html` in the current directory that can be opened with any Javascript-enabled browser to see a visual representation of the RST parse tere.


## License

This code is licensed under the MIT license (see [LICENSE.txt](LICENSE.txt)).

