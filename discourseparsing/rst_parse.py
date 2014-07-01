#!/usr/bin/env python3

import logging

from discourseparsing.discourse_parsing import Parser
from discourseparsing.discourse_segmentation import (Segmenter,
                                                     extract_edus_tokens)
from discourseparsing.parse_util import SyntaxParserWrapper
from discourseparsing.tree_util import (TREE_PRINT_MARGIN,
                                        extract_preterminals,
                                        extract_converted_terminals)


def segment_and_parse(doc_dict, syntax_parser, segmenter, rst_parser):
    '''
    A method to perform syntax parsing, discourse segmentation, and RST parsing
    as necessary, given a partial document dictionary.
    See `convert_rst_discourse_tb.py` for details about document dictionaries.
    '''

    if 'syntax_trees' not in doc_dict:
        # #TODO remove this debugging
        # import os
        # from nltk.tree import ParentedTree
        # if os.path.exists('tmp_trees'):
        #     with open('tmp_trees') as f:
        #         trees = [ParentedTree(line.strip()) for line in f]
        # else:
        #     trees = syntax_parser.parse_document(doc_dict['raw_text'])
        #     with open('tmp_trees', 'w') as f:
        #         for t in trees:
        #             print(t.pprint(TREE_PRINT_MARGIN), file=f)
        trees = syntax_parser.parse_document(doc_dict['raw_text'])
        doc_dict['syntax_trees'] = [t.pprint(margin=TREE_PRINT_MARGIN)
                                    for t in trees]
        preterminals = [extract_preterminals(t) for t in trees]
        doc_dict['token_tree_positions'] = [[x.treeposition() for x in
                                             preterminals_sentence]
                                            for preterminals_sentence
                                            in preterminals]
        doc_dict['tokens'] = [extract_converted_terminals(t) for t in trees]
        doc_dict['pos_tags'] = [[x.label() for x in preterminals_sentence]
                                for preterminals_sentence in preterminals]

    if 'edu_start_indices' not in doc_dict:
        segmenter.segment_document(doc_dict)

    edu_tags = extract_edus_tokens(doc_dict['edu_start_indices'],
                                   doc_dict['pos_tags'])
    edu_tokens = extract_edus_tokens(doc_dict['edu_start_indices'],
                                     doc_dict['tokens'])

    tagged_edus = []
    for (tags, tokens) in zip(edu_tags, edu_tokens):
        tagged_edus.append(list(zip(tokens, tags)))

    return rst_parser.parse(tagged_edus)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_files',
                        nargs='+',
                        help='A document to segment and parse.',
                        type=argparse.FileType('r'))
    parser.add_argument('-g', '--segmentation_model',
                        help='Path to segmentation model.')
    parser.add_argument('-p', '--parsing_model',
                        help='Path to RST parsing model.')
    parser.add_argument('-a', '--max_acts',
                        help='Maximum number of actions for...?',
                        type=int, default=1)
    parser.add_argument('-n', '--n_best',
                        help='Number of parses to return', type=int, default=1)
    parser.add_argument('-s', '--max_states',
                        help='Maximum number of states to retain for \
                              best-first search',
                        type=int, default=1)
    parser.add_argument('-z', '--zpar_directory', default='zpar')
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
                    n_best=args.n_best)
    parser.load_model(args.parsing_model)

    for input_file in args.input_files:
        doc = input_file.read().strip()
        logger.debug('Parsing: %s', doc)
        doc_dict = {"raw_text": doc}

        complete_trees = segment_and_parse(doc_dict, syntax_parser, segmenter,
                                           parser)

        for tree in complete_trees:
            print("{}\t{}".format(tree["tree"].pprint(margin=TREE_PRINT_MARGIN),
                                  tree["score"]))

if __name__ == '__main__':
    main()
