#!/usr/bin/env python

"""
Script to evaluate the RST parser.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import json
import logging
from collections import Counter
from operator import itemgetter

from nltk.tree import ParentedTree

from .collapse_rst_labels import collapse_rst_labels
from .discourse_parsing import Parser
from .discourse_segmentation import Segmenter, extract_edus_tokens
from .parse_util import SyntaxParserWrapper
from .rst_parse import segment_and_parse


def _extract_spans(doc_id, edu_tokens_lists, tree):
    """
    Compute the token spans for each subtree in the RST tree.

    Parameters
    ----------
    doc_id : dict
        A dictionary representing the document.
    edu_tokens_lists : list
        List of list of tokens, one list for each EDU in the document.
    tree : nltk.tree.ParentedTree
        The given RST tree.

    Returns
    -------
    spans : set
        Set of (document_id, subtree label, span start index, span end index)
        4-tuples.
    """
    # precompute the token indices for each EDU
    edu_token_spans = []
    prev_end = -1
    for edu_tokens_list in edu_tokens_lists:
        start = prev_end + 1
        end = start + len(edu_tokens_list) - 1
        edu_token_spans.append((start, end))
        prev_end = end
    # TODO: are just the start and end indices sufficient if there are same-unit
    # relations, where some other EDU (e.g., attribution) might be in the
    # middle?  Or should we use a list of tokens instead of just (start, end)
    # indices?

    # now compute the token spans for each subtree in the RST tree,
    # whose leaves are indices into the list of EDUs
    res = set()
    for subtree in tree.subtrees():
        if subtree.label() == "text" or subtree.label() == "ROOT":
            # TODO: should the nodes immediately above the text nodes be skipped
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
    """
    Compute Precision/Recall/F1 score for the predicted tuples.

    Parameters
    ----------
    gold_tuples : set
        Set of gold-standard 4-tuples.
    pred_tuples : set
        Set of predicted 4-tuples.

    Returns
    -------
    scores : tuple
        A 3-tuple containing (precision, recall, f1).
    """
    precision = float(len(gold_tuples & pred_tuples)) / len(pred_tuples)
    recall = float(len(gold_tuples & pred_tuples)) / len(gold_tuples)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_rst_eval_results(pred_edu_tokens_lists,
                             pred_trees,
                             gold_edu_tokens_lists,
                             gold_trees):
    """
    Compute the full set of RST evaluation results.

    The full set of evaluation results includes:
    - P/R/F for spans with full labels
    - P/R/F for spans with nuclearity labels
    - P/R/F for unlabeled spans

    Parameters
    ----------
    pred_edu_tokens_lists : list
        list of lists of predicted tokens, one per EDU.
    pred_trees : list
        list of predicted RST trees.
    gold_edu_tokens_lists : list
        list of lists of gold-standard tokens, one per EDU.
    gold_trees : list
        list of gold-standard RST trees.

    Returns
    -------
    results : dict
        A dictionary containing all 9 evaluations statistics.
    """
    # Extract sets of labeled spans for the gold and predicted trees.
    pred_tuples = set()
    for i, (edu_tokens_list, tree) in enumerate(zip(pred_edu_tokens_lists,
                                                    pred_trees)):
        pred_tuples |= _extract_spans(i, edu_tokens_list, tree)
    gold_tuples = set()
    for i, (edu_tokens_list, tree) in enumerate(zip(gold_edu_tokens_lists,
                                                    gold_trees)):
        gold_tuples |= _extract_spans(i, edu_tokens_list, tree)

    # (1) compute p/r/f1 for labeled spans
    (labeled_precision,
     labeled_recall,
     labeled_f1) = compute_p_r_f1(gold_tuples, pred_tuples)

    # print out false negatives/positives
    false_positives = Counter([x[1] for x in pred_tuples - gold_tuples])
    false_positives = sorted(false_positives.items(), key=itemgetter(1))
    logging.info(f"false positives: {false_positives}")

    false_negatives = Counter([x[1] for x in gold_tuples - pred_tuples])
    false_negatives = sorted(false_negatives.items(), key=itemgetter(1))
    logging.info(f"false negatives: {false_negatives}")

    # (2) compute p/r/f1 for spans with nuclearity labels
    gold_tuples = {(tup[0], tup[1].split(':')[0], tup[2], tup[3])
                   for tup in gold_tuples}
    pred_tuples = {(tup[0], tup[1].split(':')[0], tup[2], tup[3])
                   for tup in pred_tuples}
    (nuc_precision,
     nuc_recall,
     nuc_f1) = compute_p_r_f1(gold_tuples, pred_tuples)

    # (3) compute p/r/f1 for just spans and no labels
    gold_tuples = {(tup[0], tup[2], tup[3])
                   for tup in gold_tuples}
    pred_tuples = {(tup[0], tup[2], tup[3])
                   for tup in pred_tuples}
    span_precision, span_recall, span_f1 = compute_p_r_f1(gold_tuples,
                                                          pred_tuples)

    # create a dictionary with all the evaluation statistics
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


def predict_rst_trees_for_eval(syntax_parser,
                               segmenter,
                               rst_parser,
                               eval_data,
                               use_gold_syntax=True):
    """
    Predict RST trees for evaluation.

    Parameters
    ----------
    syntax_parser : SyntaxParserWrapper
        An instance of the syntactic parser wrapper.
    segmenter : Segmenter
        An instance of the discourse segmenter.
    rst_parser : Parser
        An instance of the RST parser.
    eval_data : str
        The development or evaluation set JSON file.
        (see ``convert_rst_discourse_tb.py`` for more details)
    use_gold_syntax : bool, optional
        Use gold-standard constituency trees if ``True``. If ``False``,
        use ZPar to generate the constituency trees before predicting
        the RST tree.
        Defaults to ``True``.

    Returns
    -------
    output : list
        List of (predicted EDU token lists, predicted RST trees, gold-standard
        EDU token lists, gold-standard RST trees) 4-tuples.
    """
    # initialize the lists to return
    pred_edu_tokens_lists = []
    pred_trees = []
    gold_edu_tokens_lists = []
    gold_trees = []

    # for each document in the evaluation set
    for doc_dict in eval_data:
        logging.info(f"processing {doc_dict['path_basename']}")
        gold_edu_token_list = extract_edus_tokens(doc_dict['edu_start_indices'],
                                                  doc_dict['tokens'])
        gold_edu_tokens_lists.append(gold_edu_token_list)

        # collapse the RST labels to use the coarse relations
        # that the parser produces
        gold_tree = ParentedTree.fromstring(doc_dict['rst_tree'])
        collapse_rst_labels(gold_tree)
        gold_trees.append(gold_tree)

        # TODO: when not using gold syntax, should the script still use gold
        # standard tokens?

        # remove gold standard trees or EDU boundaries if evaluating
        # using automatic preprocessing
        if not use_gold_syntax:
            # TODO: will merging the EDU strings here to make the raw_text
            # variable produce the appropriate eval result when not using gold
            # standard trees?
            doc_dict["raw_text"] = ' '.join(doc_dict["edu_strings"])
            del doc_dict["syntax_trees"]
            del doc_dict["token_tree_positions"]
            del doc_dict["tokens"]
            del doc_dict["pos_tags"]
        if segmenter is not None:
            del doc_dict["edu_start_indices"]

        # predict the RST tree for this document
        tokens, trees = segment_and_parse(doc_dict,
                                          syntax_parser,
                                          segmenter,
                                          rst_parser)
        pred_trees.append(next(trees)["tree"])
        pred_edu_tokens_lists.append(tokens)

    return (pred_edu_tokens_lists,
            pred_trees,
            gold_edu_tokens_lists,
            gold_trees)


def predict_and_evaluate_rst_trees(syntax_parser,
                                   segmenter,
                                   rst_parser,
                                   eval_data,
                                   use_gold_syntax=True):
    """
    Predict and evaluate RST trees.

    Parameters
    ----------
    syntax_parser : SyntaxParserWrapper
        An instance of the syntactic parser wrapper.
    segmenter : Segmenter
        An instance of the discourse segmenter.
    rst_parser : Parser
        An instance of the RST parser.
    eval_data : str
        The development or evaluation set JSON file.
        (see ``convert_rst_discourse_tb.py`` for more details)
    use_gold_syntax : bool, optional
        Use gold-standard constituency trees if ``True``. If ``False``,
        use ZPar to generate the constituency trees before predicting
        the RST tree.
        Defaults to ``True``.

    Returns
    -------
    results : dict
        A dictionary containing all 9 evaluations statistics.
        (see ``compute_rst_eval_results`` for details)
    """
    (pred_edu_tokens_lists,
     pred_trees,
     gold_edu_tokens_lists,
     gold_trees) = predict_rst_trees_for_eval(syntax_parser,
                                              segmenter,
                                              rst_parser,
                                              eval_data,
                                              use_gold_syntax=use_gold_syntax)

    res = compute_rst_eval_results(pred_edu_tokens_lists,
                                   pred_trees,
                                   gold_edu_tokens_lists,
                                   gold_trees)
    return res


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("evaluation_set",
                        help="The dev or test set JSON file",
                        type=argparse.FileType('r'))
    parser.add_argument("-g",
                        "--segmentation_model",
                        help="Path to the segmentation model. If not specified, "
                             "then gold EDUs will be used.",
                        default=None)
    parser.add_argument("-p",
                        "--parsing_model",
                        help="Path to the RST parsing model.",
                        required=True)
    parser.add_argument("-z",
                        "--zpar_directory",
                        default='zpar')
    parser.add_argument("-t",
                        "--use_gold_syntax",
                        help="If specified, then gold PTB syntax trees will "
                        "be used.",
                        action="store_true")
    parser.add_argument("-a",
                        "--max_acts",
                        help="Maximum number of actions to perform on each state",
                        type=int,
                        default=1)
    parser.add_argument("-s",
                        "--max_states",
                        help="Maximum number of states to retain for "
                        "best-first search",
                        type=int,
                        default=1)
    parser.add_argument("-v",
                        "--verbose",
                        help="Print more status information. For every "
                             "additional time this flag is specified, "
                             "output gets more verbose.",
                        default=0,
                        action="count")
    args = parser.parse_args()
    assert args.use_gold_syntax or args.segmentation_model

    # convert verbose flag to logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]

    # format warnings more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)
    logger = logging.getLogger(__name__)

    # read the models
    logger.info('Loading models')

    # TODO add port, host
    syntax_parser = SyntaxParserWrapper(zpar_model_directory=args.zpar_directory) if not args.use_gold_syntax else None
    segmenter = Segmenter(args.segmentation_model) if args.segmentation_model else None
    rst_parser = Parser(max_acts=args.max_acts,
                        max_states=args.max_states,
                        n_best=1)
    rst_parser.load_model(args.parsing_model)

    # read in the evaluation data
    eval_data = json.load(args.evaluation_set)

    # make predictions and get the evaluation results
    results = predict_and_evaluate_rst_trees(syntax_parser,
                                             segmenter,
                                             rst_parser,
                                             eval_data,
                                             use_gold_syntax=args.use_gold_syntax)
    print(json.dumps(sorted(results.items())))


if __name__ == "__main__":
    main()
