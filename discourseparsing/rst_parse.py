#!/usr/bin/env python3

import logging
import re
from collections import defaultdict

from discourseparsing.discourse_parsing import Parser


def edus_for_doc(doc):
    '''
    Split the document into edus, one edu per line (with POS tags)
    e.g., This/DT is/VBZ a/DT test/NN ./."

    :todo:  change this to read in the JSON format that also includes
            PTB trees.
    '''
    edus = []
    for edu_str in doc.split("\n"):
        edu = []
        # Don't split on all whitespace because of crazy characters
        for tagged_token in edu_str.strip().split(' '):
            slash_idx = tagged_token.rindex('/')
            edu.append((tagged_token[:slash_idx],
                        tagged_token[slash_idx + 1:]))
        edus.append(edu)
    return edus


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file',
                        help='file to parse, with one EDU per line, with POS \
                              tags (e.g., "This/DT is/VBZ a/DT test/NN ./.").',
                        type=argparse.FileType('r'))
    parser.add_argument('-m', '--model_file',
                        help='Path to model file.',
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

    # read the model
    logger.info('Loading model')
    weights = defaultdict(dict)
    for line in args.model_file:
        parts = line.strip().split()
        weights[parts[0]][parts[1]] = float(parts[2])
    parser.set_weights(weights)

    data = args.input_file.read().strip()
    docs = re.split(r'\n\n+', data)

    for doc in docs:
        doc_edus = edus_for_doc(doc)
        logger.debug('Parsing %s', doc_edus)
        complete_trees = parser.parse(edus_for_doc(doc))

        if args.n_best > 1:
            for tree in complete_trees:
                print(tree["score"])
                print("(ROOT {})".format(tree["tree"].pprint()))
            print()
        else:
            print("(ROOT {})".format(complete_trees[0]["tree"].pprint()))


if __name__ == '__main__':
    main()
