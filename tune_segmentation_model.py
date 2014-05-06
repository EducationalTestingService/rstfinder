#!/usr/bin/env python3

import re
import argparse
import subprocess
import shlex
import os

from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', help='The path to the training set .tsv file for CRF++')
    parser.add_argument('dev_path', help='The path to the development set .tsv file for CRF++')
    parser.add_argument('model_path_prefix', help='The path prefix for where the models should be stored.  Multiple files will be saved, for different hyperparameter settings.')
    args = parser.parse_args()

    best_f1 = -1
    best_precision = -1
    best_recall = -1
    best_C = None
    best_model_path = None

    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model_path = "{}.C{}".format(args.model_path_prefix, C)
        subprocess.call(shlex.split('crf_learn segmentation_crfpp_template.txt {} {} -c {}'.format(args.train_path, model_path, C)))
        crf_test_output = subprocess.check_output(shlex.split('crf_test -m {} {}'.format(model_path, args.dev_path))).decode('utf-8')
        output_split = [re.split(r'\t', x)[-2:] for x in re.split(r'\n+', crf_test_output) if x.strip()]
        gold = [1 if x[0] == 'B-EDU' else 0 for x in output_split]
        pred = [1 if x[1] == 'B-EDU' else 0 for x in output_split]

        f1 = f1_score(gold, pred)
        precision = precision_score(gold, pred)
        recall = recall_score(gold, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_C = C
            best_model_path = model_path

        print("model path = {}".format(model_path))
        print("C = {}".format(C))
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
