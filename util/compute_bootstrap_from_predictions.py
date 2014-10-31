#!/usr/bin/env python3.3

'''
This is a script for computing bootstrap confidence intervals
around RST parser evaluation results.

Note: this could be extended to compute CIs for the difference in performance
between two systems.
'''

import json
import logging

import numpy as np
import scikits.bootstrap as boot

from discourseparsing.discourse_parsing import Parser
from discourseparsing.rst_eval import (compute_rst_eval_results,
                                       predict_rst_trees_for_eval)


def make_score_func(metric_name):
    def score_func(data):
        return compute_rst_eval_results(data[:, 0], data[:, 1],
                                        data[:, 2], data[:, 3])[metric_name]
    return score_func


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_set',
                        help='The dev or test set JSON file',
                        type=argparse.FileType('r'))
    parser.add_argument('-p', '--parsing_model',
                        help='Path to RST parsing model.',
                        required=True)
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    parser.add_argument('--metric_name', help='name of metric to use',
                        choices=["labeled_precision",
                                 "labeled_recall",
                                 "labeled_f1",
                                 "nuc_precision",
                                 "nuc_recall",
                                 "nuc_f1",
                                 "span_precision",
                                 "span_recall",
                                 "span_f1"],
                        required=True)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--alpha', type=float, default=0.05)
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

    rst_parser = Parser(max_acts=1, max_states=1, n_best=1)
    rst_parser.load_model(args.parsing_model)

    eval_data = json.load(args.evaluation_set)

    pred_edu_tokens_lists, pred_trees, gold_edu_tokens_lists, gold_trees = \
        predict_rst_trees_for_eval(None, None, rst_parser, eval_data)

    data = np.array(list(zip(pred_edu_tokens_lists, pred_trees,
                             gold_edu_tokens_lists, gold_trees)))

    # score without bootstrapping
    orig_score = compute_rst_eval_results(pred_edu_tokens_lists,
                                          pred_trees,
                                          gold_edu_tokens_lists,
                                          gold_trees)[args.metric_name]
    tmp_score = make_score_func(args.metric_name)(data)
    assert tmp_score == orig_score

    boot_ci_lower, boot_ci_upper = \
        boot.ci(data, make_score_func(args.metric_name),
                n_samples=args.n_samples, method='bca', alpha=args.alpha)

    print("evaluation_set: {}".format(args.evaluation_set))
    print("alpha: {}".format(args.alpha))
    print("n_samples: {}".format(args.n_samples))
    print("metric: {}".format(args.metric_name))
    print("original score: {}".format(orig_score))
    print("CI: ({}, {})".format(boot_ci_lower, boot_ci_upper))


if __name__ == '__main__':
    main()

