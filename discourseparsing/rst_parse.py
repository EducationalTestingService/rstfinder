#!/usr/bin/env python3

from discourseparsing.discourse_parsing import Parser
from collections import defaultdict
import re


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-m', '--modelpath')
    parser.add_argument('input_path', help='file to parse, with one EDU \
                         per line, with POS tags \
                         (e.g., "This/DT is/VBZ a/DT test/NN ./.").')
    parser.add_argument('--max_states', type=int, default=1)
    parser.add_argument('--max_acts', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    args = parser.parse_args()

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=args.n_best)

    if not args.train:
        # read the model
        weights = defaultdict(dict)
        with open(args.modelpath) as model_file:
            for line in model_file:
                parts = line.strip().split()
                weights[parts[0]][parts[1]] = float(parts[2])
        parser.set_weights(weights)

    with open(args.input_path) as f:
        data = f.read().strip()
        docs = re.split(r'\n\n+', data)

        for doc in docs:
            # Split the document into edus, one edu per line (with POS tags)
            # e.g., This/DT is/VBZ a/DT test/NN ./."
            # TODO change this to read in the JSON format that also includes
            # PTB trees.
            edus = []
            for edu_str in doc.split("\n"):
                edu = []
                for tagged_token in edu_str.strip().split():
                    slash_idx = tagged_token.rindex('/')
                    edu.append((tagged_token[:slash_idx],
                                tagged_token[slash_idx + 1:]))
                edus.append(edu)

            tree = parser.parse(edus)


if __name__ == '__main__':
    main()
