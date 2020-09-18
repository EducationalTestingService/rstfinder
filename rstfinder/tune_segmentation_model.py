#!/usr/bin/env python
"""
Functions to tune the CRF++ based segmentation model.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import itertools
import re
import shlex
import sys
from os.path import exists
from pathlib import Path
from subprocess import call, check_output

from sklearn.metrics import f1_score, precision_score, recall_score

from .make_segmentation_crfpp_template import make_segmentation_crfpp_template


def convert_crfpp_output(crfpp_output):
    """
    Convert CRF++ command line output.

    This function takes the command line output of CRF++ and splits it into
    one [gold_label, pred_label] list per word per sentence.

    Parameters
    ----------
    crfpp_output : str
        Command line output obtained from a CRF++ command.

    Returns
    -------
    result : list
        List of [gold_label, pred_label] per word per sentence.
    """
    res = [[re.split(r'\t', token_output)[-2:] for token_output
            in re.split(r'\n', sentence_output)]
           for sentence_output in re.split(r'\n\n+', crfpp_output.strip())]
    return res


def evaluate_segmentation_output(output_by_sent):
    """
    Evaluate the output of the segmenter.

    This function returns precision, recall, F1, a gold standard count, and a
    predicted count for the B-EDU (start of an EDU) class, which corresponds
    to the start of an EDU.  This ignores EDU boundaries at sentence boundaries
    for the evaluation, following previous work (e.g., Soricut and Marcu, 2003).

    Parameters
    ----------
    output_by_sent : list
        Per-sentence output produced by ``convert_crfpp_output()``.

    Returns
    -------
    results : tuple
        5-tuple containing (precision, recall, F1, gold standard count for
        B-EDU, predicted count for B-EDU).
    """
    output_by_sent_skip1st = [x[1:] for x in output_by_sent]
    chained_output = list(itertools.chain(*output_by_sent_skip1st))

    gold = [1 if x[0] == 'B-EDU' else 0 for x in chained_output]
    pred = [1 if x[1] == 'B-EDU' else 0 for x in chained_output]

    precision = precision_score(gold, pred)
    recall = recall_score(gold, pred)
    f1 = f1_score(gold, pred)

    return precision, recall, f1, sum(gold), sum(pred)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("train_path",
                        help="The path to the training set `.tsv` "
                             "file for CRF++")
    parser.add_argument("dev_path",
                        help="The path to the development set `.tsv` "
                             "file for CRF++")
    parser.add_argument("model_path_prefix",
                        help="The path prefix for where the models should be "
                             "stored.  Multiple files will be saved, for "
                             "different hyperparameter settings.")
    parser.add_argument("--template_path",
                        help="path to the CRF++ template for segmentation "
                             "(will be created if the file does not exist)",
                        default="segmentation_crfpp_template.txt")
    parser.add_argument("-C",
                        "--C_values",
                        help="comma-separated list of model complexity "
                             "parameter settings to evaluate.",
                        default=','.join([str(2.0 ** x) for x in range(-6, 7)]))
    args = parser.parse_args()

    # initialize the variables that will hold the best results
    # across all of the C values that will be tried
    best_f1 = -1
    best_precision = -1
    best_recall = -1
    best_C = None
    best_model_path = None
    num_sentences = None

    # create the template if it does not already exist
    if not exists(args.template_path):
        make_segmentation_crfpp_template(args.template_path)

    # try each C value - first train a model with that C value and
    # then evaluate the trained model
    C_values = [float(x) for x in args.C_values.split(',')]
    for C_value in C_values:
        model_path = f"{args.model_path_prefix}.C{C_value}"
        # get full path to `crf_learn`
        crf_learn_path = Path(sys.executable).parent / "crf_learn"
        train_cmd = f"{crf_learn_path} {args.template_path} {args.train_path} {model_path} -c {C_value}"
        call(shlex.split(train_cmd))
        test_cmd = f"crf_test -m {model_path} {args.dev_path}"
        crf_test_output = check_output(shlex.split(test_cmd)).decode("utf-8")

        # split up the output into one list per token per sentence
        output_by_sent = convert_crfpp_output(crf_test_output)

        # get the number of sentences processed
        if num_sentences is None:
            num_sentences = len(output_by_sent)
        else:
            assert num_sentences == len(output_by_sent)

        # compute the various metrics
        (precision,
         recall,
         f1,
         num_gold,
         num_pred) = evaluate_segmentation_output(output_by_sent)

        # update the best metrics if appropriate
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_C = C_value
            best_model_path = model_path

        # print out some useful information for this model
        print(f"model path = {model_path}")
        print(f"C = {C_value}")
        print(f"num gold B-EDU (not including sent. boundaries) = {num_gold}")
        print(f"num pred. B-EDU (not including sent. boundaries) = {num_pred}")
        print(f"num sentences = {num_sentences}")
        print(f"precision (B-EDU class) = {precision}")
        print(f"recall (B-EDU class) = {recall}")
        print(f"F1 (B-EDU class) = {f1}")

    # print out useful information across all models tried
    print()
    print(f"best model path = {best_model_path}")
    print(f"best C = {best_C}")
    print(f"num sentences = {num_sentences}")
    print(f"best precision (B-EDU class) = {best_precision}")
    print(f"best recall (B-EDU class) = {best_recall}")
    print(f"best F1 (B-EDU class) = {best_f1}")


if __name__ == "__main__":
    main()
