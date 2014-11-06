#!/usr/bin/env python3
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
                                          for x in range(0, 8)]))
    args = parser.parse_args()

    best_f1 = -1
    best_precision = -1
    best_recall = -1
    best_C = None
    best_model_path = None

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
        output_by_sent = [[re.split(r'\t', token_output)[-2:] for token_output
                           in re.split(r'\n', sentence_output)]
                          for sentence_output
                          in re.split(r'\n\n+', crf_test_output.strip())]

        # remove EDU boundaries at sentence boundaries
        output_by_sent = [x[1:] for x in output_by_sent]

        gold = [1 if x[0] == 'B-EDU' else 0 for x
                in itertools.chain(*output_by_sent)]
        pred = [1 if x[1] == 'B-EDU' else 0 for x
                in itertools.chain(*output_by_sent)]

        f1 = f1_score(gold, pred)
        precision = precision_score(gold, pred)
        recall = recall_score(gold, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_C = C_value
            best_model_path = model_path

        print("model path = {}".format(model_path))
        print("C = {}".format(C_value))
        print("precision (B-EDU class) = {}".format(precision))
        print("recall (B-EDU class) = {}".format(recall))
        print("F1 (B-EDU class) = {}".format(f1))

    print()
    print("best model path = {}".format(best_model_path))
    print("best C = {}".format(best_C))
    print("best precision (B-EDU class) = {}".format(best_precision))
    print("best recall (B-EDU class) = {}".format(best_recall))
    print("best F1 (B-EDU class) = {}".format(best_f1))


if __name__ == '__main__':
    main()
