#!/usr/bin/env python
# License: MIT

import argparse
import itertools
import os
import re
import shlex
import subprocess

from sklearn.metrics import f1_score, precision_score, recall_score

from discourseparsing.make_segmentation_crfpp_template \
    import make_segmentation_crfpp_template


def convert_crfpp_output(crfpp_output):
    '''
    Takes the command line output of CRF++, splits it into one
    [gold_label, pred_label] list per word per sentence.
    '''
    res = [[re.split(r'\t', token_output)[-2:] for token_output
            in re.split(r'\n', sentence_output)]
           for sentence_output in re.split(r'\n\n+', crfpp_output.strip())]
    return res


def evaluate_segmentation_output(output_by_sent):
    '''
    Returns precision, recall, F1, a gold standard count, and a predicted count
    for the B-EDU (start of an EDU) class, which corresponds to the start of
    an EDU.  This ignores EDU boundaries at sentence boundaries for the
    evaluation, following previous work (e.g., Soricut and Marcu, 2003).
    '''

    output_by_sent_skip1st = [x[1:] for x in output_by_sent]
    chained_output = list(itertools.chain(*output_by_sent_skip1st))

    gold = [1 if x[0] == 'B-EDU' else 0 for x in chained_output]
    pred = [1 if x[1] == 'B-EDU' else 0 for x in chained_output]

    precision = precision_score(gold, pred)
    recall = recall_score(gold, pred)
    f1 = f1_score(gold, pred)

    return precision, recall, f1, sum(gold), sum(pred)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_path',
                        help='The path to the training set .tsv file for' +
                        ' CRF++')
    parser.add_argument('dev_path',
                        help='The path to the development set .tsv file for' +
                        ' CRF++')
    parser.add_argument('model_path_prefix',
                        help='The path prefix for where the models should be' +
                        ' stored.  Multiple files will be saved, for' +
                        ' different hyperparameter settings.')
    parser.add_argument('--template_path',
                        help='path to the CRF++ template for segmentation ' +
                        '(this will be created if the file does not exist)',
                        default='segmentation_crfpp_template.txt')
    parser.add_argument('-C', '--C_values',
                        help='comma-separated list of model complexity ' +
                        'parameter settings to evaluate.',
                        default=','.join([str(2.0 ** x)
                                          for x in range(-6, 7)]))
    args = parser.parse_args()

    best_f1 = -1
    best_precision = -1
    best_recall = -1
    best_C = None
    best_model_path = None
    num_sentences = None

    if not os.path.exists(args.template_path):
        make_segmentation_crfpp_template(args.template_path)

    C_values = [float(x) for x in args.C_values.split(',')]
    for C_value in C_values:
        model_path = "{}.C{}".format(args.model_path_prefix, C_value)
        subprocess.call(shlex.split(
            'crf_learn {} {} {} -c {}'.format(args.template_path,
                                              args.train_path,
                                              model_path,
                                              C_value)))
        crf_test_output = subprocess.check_output(shlex.split(
            'crf_test -m {} {}'.format(model_path, args.dev_path))) \
            .decode('utf-8')

        # split up the output into one list per token per sentence
        output_by_sent = convert_crfpp_output(crf_test_output)

        if num_sentences is None:
            num_sentences = len(output_by_sent)
        else:
            assert num_sentences == len(output_by_sent)

        precision, recall, f1, num_gold, num_pred = \
            evaluate_segmentation_output(output_by_sent)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_C = C_value
            best_model_path = model_path

        print("model path = {}".format(model_path))
        print("C = {}".format(C_value))
        print("num gold B-EDU (not including sent. boundaries) = {}"
              .format(num_gold))
        print("num pred. B-EDU (not including sent. boundaries) = {}"
              .format(num_pred))
        print("num sentences = {}".format(num_sentences))
        print("precision (B-EDU class) = {}".format(precision))
        print("recall (B-EDU class) = {}".format(recall))
        print("F1 (B-EDU class) = {}".format(f1))

    print()
    print("best model path = {}".format(best_model_path))
    print("best C = {}".format(best_C))
    print("num sentences = {}".format(num_sentences))
    print("best precision (B-EDU class) = {}".format(best_precision))
    print("best recall (B-EDU class) = {}".format(best_recall))
    print("best F1 (B-EDU class) = {}".format(best_f1))


if __name__ == '__main__':
    main()
