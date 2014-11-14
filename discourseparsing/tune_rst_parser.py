#!/usr/bin/env python3
# License: MIT

'''
A script to train an RST parsing model. This takes a JSON-formatted training
set created by `convert_rst_discourse_tb.py`, trains a model, and saves the
model in a user-specified location.
'''

from collections import Counter
import logging
import os
import json
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from skll.experiments import run_configuration
from skll.learner import Learner
from nltk.tree import ParentedTree

from discourseparsing.discourse_parsing import Parser
from discourseparsing.extract_actions_from_trees import extract_parse_actions
from discourseparsing.collapse_rst_labels import collapse_rst_labels
from discourseparsing.rst_eval import predict_and_evaluate_rst_trees


def train_rst_parsing_model(working_path, model_path, parameter_settings):
    '''
    parameter_settings is a dict of scikit-learn hyperparameter settings
    '''

    C_value = parameter_settings['C']
    working_subdir = os.path.join(working_path, 'C{}'.format(C_value))
    assert not os.path.exists(working_subdir)
    os.makedirs(working_subdir)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    learner_name = 'LogisticRegression'
    fixed_parameters = [{'random_state': 123456789, 'penalty': 'l1',
                         'C': C_value}]

    # Make the SKLL config file.
    cfg_dict = {"General": {"task": "train",
                            "experiment_name": "rst_parsing"},
                "Input": {"train_location": working_path,
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

    # write config file
    cfg_path = os.path.join(working_subdir, 'rst_parsing.cfg')
    cfg = ConfigParser()
    for section_name, section_dict in list(cfg_dict.items()):
        cfg.add_section(section_name)
        for key, val in section_dict.items():
            cfg.set(section_name, key, val)

    assert not os.path.exists(cfg_path)
    with open(cfg_path, 'w') as config_file:
        cfg.write(config_file)

    # run SKLL
    run_configuration(cfg_path)

    # make the model smaller/faster
    minimize_model(model_path,
                   'rst_parsing_all_feats_LogisticRegression.model')


def minimize_model(model_path, model_name):
    '''
    This function minimizes the model by removing information about features
    that get weights of 0.
    '''

    model = Learner.from_file(os.path.join(model_path, model_name))
    # Take out coefficients for features that are 0 for all classes.
    nonzero_feat_mask = ~np.all(model.model.coef_ == 0, axis=0)
    model.model.coef_ = model.model.coef_[:, nonzero_feat_mask]
    # Remove the extra words from the feat vectorizer.
    model.feat_vectorizer.restrict(nonzero_feat_mask)
    # Refit the feature selector to expect the correct size matrices.
    model.feat_selector.fit(np.ones((1, model.model.coef_.shape[1])))
    # Make the feature vectorizer return dense matrices (this is a bit faster).
    model.feat_vectorizer.set_params(sparse=False)
    # Delete the raw_coef_ attribute that sklearn *only* uses when training.
    model.model.raw_coef_ = None
    # Save the minimized model.
    model.save(os.path.join(model_path, model_name))


def train_and_eval_model(working_path, model_path, eval_data, C):
    parameter_settings = {'C': C}
    logging.info('Training model with C = {}'.format(C))
    model_path = '{}.C{}'.format(model_path, C)

    logging.info('Evaluating model with C = {}'.format(C))
    train_rst_parsing_model(working_path, model_path, parameter_settings)
    rst_parser = Parser(1, 1, 1)
    rst_parser.load_model(model_path)
    results = predict_and_evaluate_rst_trees(None, None,
                                             rst_parser, eval_data,
                                             use_gold_syntax=True)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_file',
                        help='Path to JSON training file.',
                        type=argparse.FileType('r'))
    parser.add_argument('eval_file',
                        help='Path to JSON dev or test file for ' +
                        'tuning/evaluation.',
                        type=argparse.FileType('r'))
    parser.add_argument('model_path',
                        help='Prefix for the path to where the model should be'
                        ' stored.  A suffix with the C value will be added.')
    parser.add_argument('-w', '--working_path',
                        help='Path to where intermediate files should be ' +
                        'stored', default='working')
    parser.add_argument('-C', '--C_values',
                        help='comma-separated list of model complexity ' +
                        'parameter settings to evaluate.',
                        default=','.join([str(2.0 ** x)
                                          for x in range(-4, 5)]))
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    parser.add_argument('-s', '--single_process', action='store_true',
                        help='Run in a single process for all hyperparameter' +
                        ' grid points, to simplify debugging.')
    args = parser.parse_args()

    if os.path.exists(args.working_path):
        raise IOError("{} already exists.  Stopping here to avoid the "
                      "possibility of overwriting files that are currently "
                      "being used.".format(args.working_path))
    os.makedirs(args.working_path)

    parser = Parser(1, 1, 1)

    # Convert verbose flag to actually logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]
    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)
    logger = logging.getLogger(__name__)

    logger.info('Extracting examples')
    train_data = json.load(args.train_file)
    eval_data = json.load(args.eval_file)

    train_examples = []

    for doc_dict in train_data:
        path_basename = doc_dict['path_basename']
        logging.info('Extracting examples for {}'.format(path_basename))
        tree = ParentedTree.fromstring(doc_dict['rst_tree'])
        collapse_rst_labels(tree)
        actions = extract_parse_actions(tree)

        for i, (action_str, feats) in \
                enumerate(parser.parse(doc_dict, gold_actions=actions)):
            example_id = "{}_{}".format(path_basename, i)
            example = {"x": Counter(feats), "y": action_str, "id": example_id}
            train_examples.append(example)
            # print("{} {}".format(action_str, " ".join(feats)))

    # train and evaluate a model for each value of C
    best_labeled_f1 = -1.0
    best_C = None

    # train and evaluate models with different C values in parallel
    C_values = [float(x) for x in args.C_values.split(',')]
    partial_train_and_eval_model = partial(train_and_eval_model,
                                           args.working_path, args.model_path,
                                           eval_data)

    # Make the SKLL jsonlines feature file
    train_path = os.path.join(args.working_path, 'rst_parsing.jsonlines')
    with open(train_path, 'w') as train_file:
        for example in train_examples:
            train_file.write('{}\n'.format(json.dumps(example)))

    if args.single_process:
        all_results = [partial_train_and_eval_model(C_value)
                       for C_value in C_values]
    else:
        n_workers = len(C_values)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            all_results = executor.map(partial_train_and_eval_model, C_values)

    for C_value, results in zip(C_values, all_results):
        results["C"] = C_value
        print(json.dumps(sorted(results.items())))
        if results["labeled_f1"] > best_labeled_f1:
            best_labeled_f1 = results["labeled_f1"]
            best_C = C_value

    print("best labeled F1 = {}, with C = {}".format(best_labeled_f1, best_C))


if __name__ == '__main__':
    main()
