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

from skll.experiments import run_configuration
from nltk.tree import ParentedTree

from discourseparsing.discourse_parsing import Parser
from discourseparsing.extract_actions_from_trees import extract_parse_actions
from discourseparsing.collapse18 import collapse_rst_labels
from discourseparsing.segment_document import extract_edus_tokens


def train_rst_parsing_model(train_examples, model_path, working_path):
    if not os.path.exists(working_path):
        os.mkdir(working_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    learner_name = 'LogisticRegression'
    param_grid_list = [{'C': [10.0 ** x for x in range(-2, 3)]}]
    #param_grid_list = [{'C': [1.0]}]
    grid_objective = 'f1_score_macro'
    fixed_parameters = [{'random_state': 123456789, 'penalty': 'l2'}]

    # Make the SKLL jsonlines feature file
    train_dir = working_path
    train_path = os.path.join(train_dir, 'rst_parsing.jsonlines')
    with open(train_path, 'w') as train_file:
        for example in train_examples:
            train_file.write('{}\n'.format(json.dumps(example)))

    # Make the SKLL config file.
    cfg_dict = {"General": {"task": "train",
                            "experiment_name": "rst_parsing"},
                "Input": {"train_location": train_dir,
                          "ids_to_floats": "False",
                          "featuresets": json.dumps([["rst_parsing"]]),
                          "featureset_names": json.dumps(["rst_parsing"]),
                          "suffix": '.jsonlines',
                          "fixed_parameters": json.dumps(fixed_parameters),
                          "learners": json.dumps([learner_name])},
                "Tuning": {"feature_scaling": "none",
                           "grid_search": "True",
                           "min_feature_count": "1",
                           "objective": grid_objective,
                           "param_grids": json.dumps([param_grid_list])},
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


def extract_tagged_doc_edus(doc_dict):
    edu_start_indices = doc_dict['edu_start_indices']
    res = [list(zip(edu_words, edu_tags))
           for edu_words, edu_tags
           in zip(extract_edus_tokens(edu_start_indices, doc_dict['tokens']),
                  extract_edus_tokens(edu_start_indices, doc_dict['pos_tags']))]
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_file',
                        help='Path to JSON training file.',
                        type=argparse.FileType('r'))
    parser.add_argument('model_path',
                        help='Path to where the model should be stored')
    parser.add_argument('-w', '--working_path',
                        help='Path to where intermediate files should be stored (defaults to "working" in the current directory)',
                        default='working')  # TODO is there a better default location?  e.g., /tmp?
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

    examples = []

    for doc_dict in train_data:
        path_basename = doc_dict['path_basename']
        logging.info('Extracting examples for {}'.format(path_basename))

        doc_edus = extract_tagged_doc_edus(doc_dict)
        tree = ParentedTree(doc_dict['rst_tree'])

        collapse_rst_labels(tree)

        actions = ["{}:{}".format(act.type, act.label)
                   for act in extract_parse_actions(tree)]
        logger.debug('Extracting features for %s with actions %s',
                     doc_edus, actions)

        for i, (action_str, feats) in \
                enumerate(parser.parse(doc_edus, gold_actions=actions)):
            example_id = "{}_{}".format(path_basename, i)
            example = {"x": Counter(feats), "y": action_str, "id": example_id}
            examples.append(example)
            # print("{} {}".format(action_str, " ".join(feats)))

    logger.info('Training model')
    train_rst_parsing_model(examples, args.model_path, working_path=args.working_path)


if __name__ == '__main__':
    main()
