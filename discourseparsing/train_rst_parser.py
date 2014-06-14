#!/usr/bin/env python3

import logging
import re
import json
from collections import defaultdict

from discourseparsing.discourse_parsing import Parser


def gold_action_gen(action_file, edus):
    '''
    Given an "actionseq" file and a list of EDUs, this will generate gold
    parser actions and a subset of the EDUs that those actions go with.
    '''
    doc_for_actions = []
    for line in action_file:
        line = line.strip()
        # ignore blanks
        if line:
            # action line
            if line.startswith('S:'):
                actions = line.split(' ')
                yield doc_for_actions, actions
            # EDU indices line
            else:
                doc_for_actions = []
                for slash_str in line.split(' '):
                    idx = int(slash_str.split('/')[0])
                    doc_for_actions.append(edus[idx - 1])


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
    import pdb;pdb.set_trace()
    doc_edus = None #TODO
    actions = None #TODO

    for doc_edus, actions in gold_action_gen(args.train_file, edus):
        logger.debug('Extracting features for %s with actions %s',
                     doc_edus, actions)
        parser.parse(doc_edus, gold_actions=actions)


if __name__ == '__main__':
    main()
