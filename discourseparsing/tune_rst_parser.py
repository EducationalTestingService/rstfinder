#!/usr/bin/env python3

'''
A script to train an RST parsing model.
This takes a JSON-formatted training set created by `convert_rst_discourse_tb.py`,
trains a model, and saves the model in a user-specified location.
'''

from collections import Counter
import logging
import os
import json
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from skll.experiments import run_configuration
from nltk.tree import ParentedTree

from discourseparsing.discourse_parsing import Parser
from discourseparsing.extract_actions_from_trees import extract_parse_actions
from discourseparsing.collapse_rst_labels import collapse_rst_labels
from discourseparsing.rst_eval import predict_and_evaluate_rst_trees


def train_rst_parsing_model(train_examples, model_path, working_path,
                            parameter_settings):
    '''
    parameter_settings is a dict of scikit-learn hyperparameter settings
    '''
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    learner_name = 'LogisticRegression'
    fixed_parameters = [{'random_state': 123456789, 'penalty': 'l1',
                         'C': parameter_settings['C']}]

    # Make the SKLL jsonlines feature file
    train_path = os.path.join(working_path, 'rst_parsing.jsonlines')
    with open(train_path, 'w') as train_file:
        for example in train_examples:
            train_file.write('{}\n'.format(json.dumps(example)))

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
                           "log": working_path}
               }

    # write config file
    cfg_path = os.path.join(working_path, 'rst_parsing.cfg')
    cfg = ConfigParser()
    for section_name, section_dict in list(cfg_dict.items()):
        cfg.add_section(section_name)
        for key, val in section_dict.items():
            cfg.set(section_name, key, val)
    with open(cfg_path, 'w') as config_file:
        cfg.write(config_file)

    # run SKLL
    run_configuration(cfg_path)


def train_and_eval_model(train_examples, eval_data, working_path,
                         model_path, C):
    parameter_settings = {'C': C}
    logging.info('Training model')
    model_path = '{}.C{}'.format(model_path, C)
    working_path = os.path.join(working_path, 'C{}'.format(C))
    train_rst_parsing_model(train_examples, model_path,
                            working_path=working_path,
                            parameter_settings=parameter_settings)
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
                        help='Path to JSON dev or test file for tuning/evaluation.',
                        type=argparse.FileType('r'))
    parser.add_argument('model_path',
                        help='Prefix for the path to where the model should be '
                        'stored.  A suffix with the C value will be added.')
    parser.add_argument('-w', '--working_path',
                        help='Path to where intermediate files should be stored (defaults to "working" in the current directory)',
                        default='working')  # TODO is there a better default location?  e.g., /tmp?
    parser.add_argument('-C', '--C_values',
                        help='comma-separated list of model complexity ' +
                        'parameter settings to evaluate.',
                        default=','.join([str(10.0 ** x) for x in range(-2, 3)]))
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    args = parser.parse_args()

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

    # TODO remove or comment out the following debugging command
    # train_data = train_data[:20]

    examples = []

    for doc_dict in train_data:
        path_basename = doc_dict['path_basename']
        logging.info('Extracting examples for {}'.format(path_basename))
        tree = ParentedTree(doc_dict['rst_tree'])
        collapse_rst_labels(tree)
        actions = extract_parse_actions(tree)

        for i, (action_str, feats) in \
                enumerate(parser.parse(doc_dict, gold_actions=actions)):
            example_id = "{}_{}".format(path_basename, i)
            example = {"x": Counter(feats), "y": action_str, "id": example_id}
            examples.append(example)
            # print("{} {}".format(action_str, " ".join(feats)))

    # train and evaluate a model for each value of C
    best_labeled_f1 = -1.0
    best_C = None

    # train and evaluate models with different C values in parallel
    C_values = [float(x) for x in args.C_values.split(',')]
    partial_train_and_eval_model = partial(train_and_eval_model,
                                           examples, eval_data,
                                           args.working_path,
                                           args.model_path)
    n_workers = len(C_values)
    #n_workers = 1
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        all_results = executor.map(partial_train_and_eval_model, C_values)

    # all_results = []
    # for C_value in C_values:
    #     all_results.append(partial_train_and_eval_model(C_value))

    for C_value, results in zip(C_values, all_results):
        results["C"] = C_value
        print(json.dumps(sorted(results.items())))
        if results["labeled_f1"] > best_labeled_f1:
            best_labeled_f1 = results["labeled_f1"]
            best_C = C_value

    print("best labeled F1 = {}, with C = {}".format(best_labeled_f1, best_C))


if __name__ == '__main__':
    main()
