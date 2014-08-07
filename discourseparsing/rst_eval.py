#!/usr/bin/env python3

from collections import Counter
import json
import logging
from operator import itemgetter

from nltk.tree import ParentedTree

from discourseparsing.discourse_parsing import Parser
from discourseparsing.discourse_segmentation import (Segmenter,
                                                     extract_edus_tokens)
from discourseparsing.parse_util import SyntaxParserWrapper
from discourseparsing.rst_parse import segment_and_parse
from discourseparsing.collapse_rst_labels import collapse_rst_labels


def _extract_spans(doc_id, edu_tokens_lists, tree):
    # Precompute the token indices for each EDU.
    edu_token_spans = []
    prev_end = -1
    for edu_tokens_list in edu_tokens_lists:
        start = prev_end + 1
        end = start + len(edu_tokens_list) - 1
        edu_token_spans.append((start, end))
        prev_end = end
    # TODO Are just the start and end indices sufficient if there are same-unit
    # relations, where some other EDU (e.g., attribution) might be in the
    # middle?  Or should we use a list of tokens instead of just (start, end)
    # indices?

    # Now compute the token spans for each subtree in the RST tree,
    # whose leaves are indices into the list of EDUs.
    res = set()
    for subtree in tree.subtrees():
        if subtree.label() == 'text' or subtree.label() == 'ROOT':
            # TODO should the nodes immediately above the text nodes be skipped
            # as well (they seem important for evaluating labeled spans but
            # trivial for the case of evaluating unlabeled spans)
            continue
        leaves = subtree.leaves()
        res.add((doc_id,
                 subtree.label(),
                 edu_token_spans[int(leaves[0])][0],
                 edu_token_spans[int(leaves[-1])][1]))
    return res


def compute_p_r_f1(gold_tuples, pred_tuples):
    precision = float(len(gold_tuples & pred_tuples)) / len(pred_tuples)
    recall = float(len(gold_tuples & pred_tuples)) / len(gold_tuples)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_rst_eval_results(pred_edu_tokens_lists, pred_trees,
                             gold_edu_tokens_lists, gold_trees):

    # Extract sets of labeled spans for the gold and predicted trees.
    pred_tuples = set()
    for i, (edu_tokens_list, tree) in enumerate(zip(pred_edu_tokens_lists,
                                                    pred_trees)):
        pred_tuples |= _extract_spans(i, edu_tokens_list, tree)
    gold_tuples = set()
    for i, (edu_tokens_list, tree) in enumerate(zip(gold_edu_tokens_lists,
                                                    gold_trees)):
        gold_tuples |= _extract_spans(i, edu_tokens_list, tree)

    # Evaluate F1 for unlabeled spans, spans with nuclearity labels,
    # and spans with full labels.

    # Compute p/r/f1 for labeled spans.
    labeled_precision, labeled_recall, labeled_f1 \
        = compute_p_r_f1(gold_tuples, pred_tuples)

    logging.info('false positives: {}'.format(
        sorted(Counter([x[1] for x in pred_tuples - gold_tuples]).items(),
               key=itemgetter(1))))
    logging.info('false negatives: {}'.format(
        sorted(Counter([x[1] for x in gold_tuples - pred_tuples]).items(),
               key=itemgetter(1))))
    # confusions = [(x[1], y[1]) for x, y in
    #               itertools.product(gold_tuples, pred_tuples)
    #               if x[0] == y[0] and x[2:3] == y[2:3] and x[1] != y[1]]
    # logging.info('confusions (x, y): {}'.format(
    #     sorted(Counter(confusions).items(),
    #            key=itemgetter(1))))

    # Compute p/r/f1 for spans + nuclearity.
    gold_tuples = {(tup[0], tup[1].split(':')[0], tup[2], tup[3])
                   for tup in gold_tuples}
    pred_tuples = {(tup[0], tup[1].split(':')[0], tup[2], tup[3])
                   for tup in pred_tuples}
    nuc_precision, nuc_recall, nuc_f1 = \
        compute_p_r_f1(gold_tuples, pred_tuples)

    # Compute p/r/f1 for just spans.
    gold_tuples = {(tup[0], tup[2], tup[3])
                   for tup in gold_tuples}
    pred_tuples = {(tup[0], tup[2], tup[3])
                   for tup in pred_tuples}
    span_precision, span_recall, span_f1 = \
        compute_p_r_f1(gold_tuples, pred_tuples)

    # Create a list of all the eval statistics.
    res = {"labeled_precision": labeled_precision,
           "labeled_recall": labeled_recall,
           "labeled_f1": labeled_f1,
           "nuc_precision": nuc_precision,
           "nuc_recall": nuc_recall,
           "nuc_f1": nuc_f1,
           "span_precision": span_precision,
           "span_recall": span_recall,
           "span_f1": span_f1}

    return res


def predict_and_evaluate_rst_trees(syntax_parser, segmenter,
                                   rst_parser, eval_data,
                                   use_gold_syntax=True):
    pred_edu_tokens_lists = []
    pred_trees = []
    gold_edu_tokens_lists = []
    gold_trees = []

    for doc_dict in eval_data:
        logging.info('processing {}...'.format(doc_dict['path_basename']))
        gold_edu_tokens_lists.append( \
            extract_edus_tokens(doc_dict['edu_start_indices'],
                                doc_dict['tokens']))

        # Collapse the RST labels to use the coarse relations that the parser
        # produces.
        gold_tree = ParentedTree.fromstring(doc_dict['rst_tree'])
        collapse_rst_labels(gold_tree)
        gold_trees.append(gold_tree)

        # TODO when not using gold syntax, should the script still use gold
        # standard tokens?

        # remove gold standard trees or EDU boundaries if evaluating
        # using automatic preprocessing
        if not use_gold_syntax:
            # TODO will merging the EDU strings here to make the raw_text
            # variable produce the appropriate eval result when not using gold
            # standard trees?
            doc_dict['raw_text'] = ' '.join(doc_dict['edu_strings'])
            del doc_dict['syntax_trees']
            del doc_dict['token_tree_positions']
            del doc_dict['tokens']
            del doc_dict['pos_tags']
        if segmenter is not None:
            del doc_dict['edu_start_indices']

        # predict the RST tree
        tokens, trees = segment_and_parse(doc_dict, syntax_parser,
                                          segmenter, rst_parser)
        pred_trees.append(next(trees)['tree'])
        pred_edu_tokens_lists.append(tokens)

    results = compute_rst_eval_results(pred_edu_tokens_lists, pred_trees,
                                       gold_edu_tokens_lists, gold_trees)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_set',
                        help='The dev or test set JSON file',
                        type=argparse.FileType('r'))
    parser.add_argument('-g', '--segmentation_model',
                        help='Path to segmentation model.  If not specified,' +
                        'then gold EDUs will be used.',
                        default=None)
    parser.add_argument('-p', '--parsing_model',
                        help='Path to RST parsing model.',
                        required=True)
    parser.add_argument('-z', '--zpar_directory', default='zpar')
    parser.add_argument('-t', '--use_gold_syntax',
                        help='If specified, then gold PTB syntax trees will' +
                        'be used.', action='store_true')
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
    assert args.use_gold_syntax or args.segmentation_model

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

    # TODO add port, host, model args
    syntax_parser = SyntaxParserWrapper() if not args.use_gold_syntax else None
    segmenter = Segmenter(args.segmentation_model) \
        if args.segmentation_model else None

    rst_parser = Parser(max_acts=args.max_acts,
                        max_states=args.max_states,
                        n_best=1)
    rst_parser.load_model(args.parsing_model)

    eval_data = json.load(args.evaluation_set)

    results = \
        predict_and_evaluate_rst_trees(syntax_parser, segmenter, rst_parser,
                                       eval_data,
                                       use_gold_syntax=args.use_gold_syntax)
    print(json.dumps(sorted(results.items())))


if __name__ == '__main__':
    main()
