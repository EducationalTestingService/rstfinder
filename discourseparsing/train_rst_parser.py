#!/usr/bin/env python3

import logging
import json

from discourseparsing.discourse_parsing import Parser
from discourseparsing.extract_actions_from_trees import extract_parse_actions
from discourseparsing.collapse18 import collapse_rst_labels
from discourseparsing.segment_document import extract_edus_tokens
from nltk.tree import ParentedTree


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
    parser.add_argument('-a', '--max_acts',
                        help='Maximum number of actions for...?',
                        type=int, default=1)
    parser.add_argument('-n', '--n_best',
                        help='Number of parses to return', type=int, default=1)
    parser.add_argument('-s', '--max_states',
                        help='Maximum number of states to retain for \
                              best-first search',
                        type=int, default=1)
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    args = parser.parse_args()

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=args.n_best)

    # Convert verbose flag to actually logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]
    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)
    logger = logging.getLogger(__name__)

    logger.info('Training model')
    # Create a giant list of all EDUs for all documents

    train_data = json.load(args.train_file)

    for doc_dict in train_data:
        logging.info(doc_dict['path_basename'])

        doc_edus = extract_tagged_doc_edus(doc_dict)
        tree = ParentedTree(doc_dict['rst_tree'])

        collapse_rst_labels(tree)

        actions = ["{}:{}".format(act.type, act.label) for act in extract_parse_actions(tree)]
        logger.debug('Extracting features for %s with actions %s',
                     doc_edus, actions)

        for action_str, feats in parser.parse(doc_edus, gold_actions=actions):
            print("{} {}".format(action_str, " ".join(feats)))

    # TODO modify the feature/label format and run a learning algorithm (e.g., with SKLL)


if __name__ == '__main__':
    main()
