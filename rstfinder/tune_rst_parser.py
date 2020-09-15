#!/usr/bin/env python
"""
Train an RST parsing model.

This script takes a JSON-formatted training set created by
``convert_rst_discourse_tb.py``, trains a model, and
saves the model in a user-specified location.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from configparser import ConfigParser
from functools import partial
from os.path import abspath, exists, join

import numpy as np
from nltk.tree import ParentedTree
from skll.experiments import run_configuration
from skll.learner import Learner

from .collapse_rst_labels import collapse_rst_labels
from .discourse_parsing import Parser
from .extract_actions_from_trees import extract_parse_actions
from .rst_eval import predict_and_evaluate_rst_trees


def train_rst_parsing_model(working_path, model_path, C):
    """
    Train an RST parsing model on pre-extracted featuers and save to disk.

    This function trains a logistic regression RST parsing model on features
    that are assumed to have already been extracted to a file called
    ``rst_parsing.jsonlines`` under ``working_path``. The  ``C``
    hyperparameter of the model is set to the value provided as input
    and the model is saved under ``model_path`` with the name
    ``rst_parsing_all_feats_LogisticRegression.model``.

    Parameters
    ----------
    working_path : str
        Path to the directory where the pre-extracted SKLL training features
        are stored.
    model_path : str
        Path to the directory where the trained model will be saved.
    C : float
        The value for the ``C`` hyperparameter value to be used
        when training the model.
    """
    # create a sub-directory under ``working_path`` to store the logs
    working_subdir = join(working_path, f"C{C}")
    os.makedirs(working_subdir, exist_ok=False)

    # create ``model_path`` unless it already exists
    os.makedirs(model_path, exist_ok=True)

    # set up the learner name and settings for use in the SKLL configuration
    learner_name = 'LogisticRegression'
    fixed_parameters = [{"random_state": 123456789,
                         "penalty": 'l1',
                         'C': C}]

    # create the SKLL config dictionary
    cfg_dict = {"General": {"task": "train",
                            "experiment_name": "rst_parsing"},
                "Input": {"train_directory": working_path,
                          "ids_to_floats": "False",
                          "featuresets": json.dumps([["rst_parsing"]]),
                          "featureset_names": json.dumps(["all_feats"]),
                          "suffix": '.jsonlines',
                          "fixed_parameters": json.dumps(fixed_parameters),
                          "learners": json.dumps([learner_name])},
                "Tuning": {"feature_scaling": "none",
                           "grid_search": "False",
                           "min_feature_count": "1"},
                "Output": {"probability": "True",
                           "models": model_path,
                           "log": working_subdir}
                }

    # save the configuration file to disk
    cfg_path = join(working_subdir, "rst_parsing.cfg")
    cfg = ConfigParser()
    for section_name, section_dict in list(cfg_dict.items()):
        cfg.add_section(section_name)
        for key, val in section_dict.items():
            cfg.set(section_name, key, val)

    assert not exists(cfg_path)
    with open(cfg_path, 'w') as config_file:
        cfg.write(config_file)

    # run SKLL to train the model
    run_configuration(cfg_path)

    # make the trained model smaller/faster by removing features
    # that get zero weights due to the L1 regularization
    prune_model(model_path, "rst_parsing_all_feats_LogisticRegression.model")


def prune_model(model_path, model_name):
    """
    Prune zero-weighted features from the given logistic regression model.

    This function makes the given model smaller by removing information
    about features that get weights of 0 due to L1-regularization. The
    model file is assumed to have the name ``model_name`` and be located
    under ``model_path``.

    **IMPORTANT**: Note that the input model file is overwritten with
    the pruned model.

    Parameters
    ----------
    model_path : str
        Path to the directory that contains the model file.
    model_name : str
        The name of the model file.
    """
    # load the Learner instance from the model file
    model = Learner.from_file(join(model_path, model_name))

    # remove coefficients for features that are 0 for all classes
    nonzero_feat_mask = ~np.all(model.model.coef_ == 0, axis=0)
    model.model.coef_ = model.model.coef_[:, nonzero_feat_mask]

    # remove the extra words from the feature vectorizer
    model.feat_vectorizer.restrict(nonzero_feat_mask)

    # refit the feature selector to expect the correctly-sized matrices
    model.feat_selector.fit(np.ones((1, model.model.coef_.shape[1])))

    # make the vectorizer return dense matrices since that is a bit faster
    model.feat_vectorizer.set_params(sparse=False)

    # delete the raw_coef_ attribute that sklearn *only* uses when training
    model.model.raw_coef_ = None

    # save the pruned model to the same file
    model.save(join(model_path, model_name))


def train_and_evaluate_model(working_path, model_path, eval_data, C_value):
    """
    Train and evaluate given RST parsing model.

    Parameters
    ----------
    working_path : str
        Path to the directory where the pre-extracted SKLL training features
        are stored.
    model_path : str
        Prefix for the directory where the trained model will be saved.
        The suffix ``.C{C_value}`` is added to create the actual directory
        name.
    eval_data : str
        Path to the JSON file containing the documents on which to
        evaluate the trained parser.
    C_value : float
        The value for the ``C`` hyperparameter to use for the
        logistic regression model.

    Returns
    -------
    results : dict
        Dictionary containing the evaluation results.
    """
    # get the full name of the model directory
    logging.info(f"Training model with C = {C_value}")
    model_path = f"{model_path}.C{C_value}"

    # train a logistic regression parsing model
    train_rst_parsing_model(working_path, model_path, C_value)

    # instantiate a Parser container to hold the model
    rst_parser = Parser(1, 1, 1)
    rst_parser.load_model(model_path)

    # evaluate the model on the given data
    logging.info(f"Evaluating model with C = {C_value}")
    results = predict_and_evaluate_rst_trees(None,
                                             None,
                                             rst_parser,
                                             eval_data,
                                             use_gold_syntax=True)
    return results


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("train_file",
                        help="Path to the JSON training data.",
                        type=argparse.FileType('r'))
    parser.add_argument("eval_file",
                        help="Path to the JSON development or test data for "
                             "tuning/evaluation.",
                        type=argparse.FileType('r'))
    parser.add_argument("model_path",
                        help="A prefix for the path where the model file "
                             "should be saved. A suffix with the C value "
                             "will be added to create the full path.")
    parser.add_argument("-w",
                        "--working_path",
                        help="Path to where intermediate files should be "
                             "stored.",
                        default=join(os.getcwd(), "working"))
    parser.add_argument("-C",
                        "--C_values",
                        help="comma-separated list of model complexity "
                             "hyperparameter values to evaluate.",
                        default=','.join([str(2.0 ** x) for x in range(-4, 5)]))
    parser.add_argument("-v",
                        "--verbose",
                        help="Print more status information. For every "
                             "additional time this flag is specified, "
                             "output gets more verbose.",
                        default=0,
                        action='count')
    parser.add_argument("-s",
                        "--single_process",
                        action='store_true',
                        help="Run all hyperparameter values in a single "
                             "process, to simplify debugging.")
    args = parser.parse_args()

    # convert given paths to absolute paths
    working_path = abspath(args.working_path)
    model_path = abspath(args.model_path)

    if exists(working_path):
        raise IOError(f"{working_path} already exists. Stopping here "
                      f"to avoid the possibility of overwriting files that "
                      f"are currently being used.")
    else:
        os.makedirs(working_path)

    # instantiate a parser container
    parser = Parser(1, 1, 1)

    # convert verbose flag to logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]

    # format warnings more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=("%(asctime)s - %(name)s - %(levelname)s - "
                                "%(message)s"),
                        level=log_level)
    logger = logging.getLogger(__name__)

    # extract the training examples
    logger.info("Extracting examples")
    train_data = json.load(args.train_file)
    eval_data = json.load(args.eval_file)
    train_examples = []

    # iterate over each document in the training data
    for doc_dict in train_data:
        path_basename = doc_dict["path_basename"]
        logging.info(f"Extracting examples for {path_basename}")
        tree = ParentedTree.fromstring(doc_dict['rst_tree'])
        collapse_rst_labels(tree)
        actions = extract_parse_actions(tree)

        # extract the training features
        parser_tuples = parser.parse(doc_dict, gold_actions=actions)
        for i, (action_str, feats) in enumerate(parser_tuples):
            example_id = f"{path_basename}_{i}"
            example = {"x": Counter(feats), "y": action_str, "id": example_id}
            train_examples.append(example)

    # save the training features to disk in SKLL jsonlines format
    train_path = join(working_path, "rst_parsing.jsonlines")
    with open(train_path, 'w') as train_file:
        for example in train_examples:
            train_file.write(f"{json.dumps(example)}\n")

    # instantiate some variables
    best_labeled_f1 = -1.0
    best_C = None

    # train and evaluate models with different C values in parallel
    C_values = [float(x) for x in args.C_values.split(',')]
    partial_trainer = partial(train_and_evaluate_model,
                              working_path,
                              model_path,
                              eval_data)

    # run in a single process or using multiple processes
    if args.single_process:
        all_results = [partial_trainer(C_value) for C_value in C_values]
    else:
        n_workers = len(C_values)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            all_results = executor.map(partial_trainer, C_values)

    # find the C value that yields the best model
    for C_value, results in zip(C_values, all_results):
        results["C"] = C_value
        print(json.dumps(sorted(results.items())))
        if results["labeled_f1"] > best_labeled_f1:
            best_labeled_f1 = results["labeled_f1"]
            best_C = C_value

    # print out the results for the best C
    print(f"best labeled F1 = {best_labeled_f1}, with C = {best_C}")


if __name__ == "__main__":
    main()
