#!/usr/bin/env python

import json
import logging

from nltk.tree import ParentedTree

from discourseparsing.extract_actions_from_trees import extract_parse_actions
from discourseparsing.discourse_segmentation import extract_edus_tokens
from discourseparsing.discourse_parsing import Parser


def test_extract_parse_actions():
    tree = ParentedTree('(ROOT (satellite:attribution (text 0)) (nucleus:span (satellite:condition (text 1)) (nucleus:span (nucleus:span (nucleus:same-unit (text 2)) (nucleus:same-unit (satellite:temporal (text 3)) (nucleus:span (text 4)))) (satellite:conclusion (text 5)))))')
    # I think the tree above would be for something
    # like this silly little example:
    # "John said that if Bob bought this excellent book,
    # then before the end of next week Bob would finish it,
    # and therefore he would be happy."

    actions = extract_parse_actions(tree)

    num_shifts = len([x for x in actions if x.type == 'S'])
    assert num_shifts == 6
    assert actions[0].type == 'S'
    assert actions[1].type == 'U'
    assert actions[1].label == 'satellite:attribution'
    assert actions[2].type == 'S'


def test_reconstruct_training_examples():
    '''
    This code goes through the training data and makes sure
    that the actions extracted from the trees can be used to
    reconstruct those trees from a list of EDUs.
    '''

    train_path = 'rst_discourse_tb_edus_TRAINING_TRAIN.json'
    with open(train_path) as f:
        data = json.load(f)

    rst_parser = Parser(max_acts=1, max_states=1, n_best=1)
    for doc_dict in data:
        tree_orig = ParentedTree(doc_dict['rst_tree'])
        actions = extract_parse_actions(tree_orig)

        edu_tags = extract_edus_tokens(doc_dict['edu_start_indices'],
                                       doc_dict['pos_tags'])
        edu_tokens = extract_edus_tokens(doc_dict['edu_start_indices'],
                                         doc_dict['tokens'])
        tagged_edus = []
        for (tags, tokens) in zip(edu_tags, edu_tokens):
            tagged_edus.append(list(zip(tokens, tags)))

        tree2 = next(rst_parser.parse(tagged_edus,
                                      gold_actions=actions,
                                      make_features=False))['tree']

        logging.info('test_reconstruct_training_examples verified tree for {}'.format(doc_dict['path_basename']))
        assert tree2 == tree_orig


if __name__ == '__main__':
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)
    test_extract_parse_actions()
    test_reconstruct_training_examples()
