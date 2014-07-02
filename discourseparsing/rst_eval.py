#!/usr/bin/env python3

import json
import logging

from nltk.tree import ParentedTree

from discourseparsing.discourse_parsing import Parser
from discourseparsing.discourse_segmentation import (Segmenter,
                                                     extract_edus_tokens)
from discourseparsing.parse_util import SyntaxParserWrapper
from discourseparsing.rst_parse import segment_and_parse


def compute_rst_eval_results(pred_edu_tokens, pred_trees, gold_edu_tokens,
                             gold_trees):
    res = None
    # TODO extract sets of labeled spans for the gold and predicted trees

    # TODO Evaluate F1 for unlabeled spans, spans with nuclearity labels, and spans with full labels 
    import pdb;pdb.set_trace()
    return res


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_set',
                        help='The dev or test set JSON file',
                        type=argparse.FileType('r'))
    parser.add_argument('-g', '--segmentation_model',
                        help='Path to segmentation model.')
    parser.add_argument('-p', '--parsing_model',
                        help='Path to RST parsing model.')
    parser.add_argument('-z', '--zpar_directory', default='zpar')
    parser.add_argument('-e', '--use_gold_edus', action='store_true')
    parser.add_argument('-e', '--use_gold_trees', action='store_true')
    parser.add_argument('-a', '--max_acts',
                        help='Maximum number of actions to perform on each ' +
                        'state', type=int, default=1)
    parser.add_argument('-s', '--max_states',
                        help='Maximum number of states to retain for ' +
                        'best-first search', type=int, default=1)
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    args = parser.parse_args()

    # Convert verbose flag to actually logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]
    # Make warnings from built-in warnings module get formatted more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)
    logger = logging.getLogger(__name__)

    # read the models
    logger.info('Loading models')
    syntax_parser = SyntaxParserWrapper(args.zpar_directory)
    segmenter = Segmenter(args.segmentation_model)

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=1)
    parser.load_model(args.parsing_model)

    eval_data = json.load(args.evaluation_set)

    pred_edu_tokens = []
    pred_trees = []
    gold_edu_tokens = []
    gold_trees = []

    for doc_dict in eval_data:
        gold_edu_tokens.append(extract_edus_tokens(doc_dict['edu_start_indices'],
                                                   doc_dict['tokens']))
        gold_trees.append(ParentedTree(doc_dict['rst_tree']))

        # remove gold standard trees or EDU boundaries if evaluating
        # using automatic preprocessing
        if not args.use_gold_trees:
            # TODO will merging the EDU strings here to make the raw_text variable produce the appropriate eval result when not using gold standard trees?
            doc_dict['raw_text'] = ' '.join(doc_dict['edu_strings'])
            del doc_dict['syntax_trees']
            del doc_dict['token_tree_positions']
            del doc_dict['tokens']
            del doc_dict['pos_tags']
        if not args.use_gold_edus:
            del doc_dict['edu_start_indices']

        # predict the RST tree
        tokens, trees = segment_and_parse(doc_dict, syntax_parser,
                                                   segmenter, parser)
        pred_trees.append(next(trees))
        pred_edu_tokens.append(tokens)

    results = compute_rst_eval_results(pred_edu_tokens, pred_trees,
                                       gold_edu_tokens, gold_trees)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()

