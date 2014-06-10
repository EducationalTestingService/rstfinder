#!/usr/bin/env python3

import re
from collections import defaultdict

from discourseparsing.discourse_parsing import Parser


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file',
                        help='file to parse, with one EDU per line, with POS \
                              tags (e.g., "This/DT is/VBZ a/DT test/NN ./.").',
                        type=argparse.FileType('r'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model_file',
                       help='Path to model file.',
                       type=argparse.FileType('r'))
    group.add_argument('-t', '--train_file',
                       help='Path to training file.',
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
    args = parser.parse_args()

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=args.n_best)

    if args.train_file is None:
        # read the model
        weights = defaultdict(dict)
        for line in args.model_file:
            parts = line.strip().split()
            weights[parts[0]][parts[1]] = float(parts[2])
        parser.set_weights(weights)

    data = args.input_file.read().strip()
    docs = re.split(r'\n\n+', data)

    for doc in docs:
        # Split the document into edus, one edu per line (with POS tags)
        # e.g., This/DT is/VBZ a/DT test/NN ./."
        # TODO change this to read in the JSON format that also includes
        # PTB trees.
        edus = []
        for edu_str in doc.split("\n"):
            edu = []
            # Don't split on all whitespace because of crazy characters
            for tagged_token in edu_str.strip().split(' '):
                slash_idx = tagged_token.rindex('/')
                edu.append((tagged_token[:slash_idx],
                            tagged_token[slash_idx + 1:]))
            edus.append(edu)

        parser.parse(edus, train_file=args.train_file)


if __name__ == '__main__':
    main()
