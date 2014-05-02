#!/usr/bin/env python3

'''
A discourse segmenter following the Base model from this paper:
Ngo Xuan Bach, Nguyen Le Minh, Akira Shimazu. 2012.
A Reranking Model for Discourse Segmentation using Subtree Features.
SIGDIAL. http://aclweb.org/anthology//W/W12/W12-1623.pdf.
'''

import argparse
import json
from nltk.tree import ParentedTree


def find_first_common_ancestor(n1, n2):
    '''
    :param n1: node in tree t
    :type n1: ParentedTree
    :param n2: node in tree t
    :type n2: ParentedTree

    Find the first common ancestor for the two nodes n1 and n2 in the same
    tree.
    '''

    # make sure we are in the same tree
    assert n1.root() == n2.root()

    # make a set of all ancestors of n1
    n1_ancestor_treepositions = set()
    n1_parent = n1.parent()
    while n1_parent is not None:
        # Note: this storing treepositions isn't
        # particularly efficient since treeposition() walks up the tree.
        # Using memory addresses like id(n1_parent)
        # would be faster, but seems potentially hazardous/confusing.
        n1_ancestor_treepositions.add(n1_parent.treeposition())
        n1_parent = n1_parent.parent()

    # find the first ancestor of n2 that is also an ancestor of n1
    n2_parent = n2.parent()
    res = None
    while n2_parent is not None:
        if n2_parent.treeposition() in n1_ancestor_treepositions:
            res = n2_parent
            break
        n2_parent = n2_parent.parent()

    assert res is not None
    return res


def find_head_preterminal(node):
    return node.leaves()[0]  # TODO implement or find an implementation of collins head rules


def parse_node_features(nodes):
    for node in nodes:
        node_head_preterminal = find_head_preterminal(node)
        yield '{}({})'.format(node.label(), node_head_preterminal[0]) if node else ""
        yield '{}({})'.format(node.label(), node_head_preterminal) if node else ""


def extract_segmentation_features(doc_dict):
    '''
    :param doc_dict: A dictionary of edu_start_indices, tokens, syntax_trees,
                token_tree_positions, and pos_tags for a document, as
                extracted by convert_rst_discourse_tb.py.
    '''
    labels = []
    feat_lists = []
    edu_starts = {(x[0], x[1]) for x in doc_dict['edu_start_indices']}
    for sent_num, (sent_tokens, tree_str, sent_tree_positions, pos_tags) in enumerate(zip(doc_dict['tokens'], doc_dict['syntax_trees'], doc_dict['token_tree_positions'], doc_dict['pos_tags'])):
        tree = ParentedTree(tree_str)
        for token_num, (token, tree_position, pos_tag) in enumerate(zip(sent_tokens, sent_tree_positions, pos_tags)):
            feats = []
            label = 'B-EDU' if (sent_num, token_num) in edu_starts else 'C-EDU'

            # TODO: all of the stuff below needs to be checked

            # POS tags and words for lexicalized parse nodes
            # from 3.2 of Bach et al., 2012.
            # preterminal node for the current word
            node_w = tree[tree_position]
            # node for the word to the right
            node_r = tree[sent_tree_positions[token_num + 1]] if token_num + 1 < len(sent_tree_positions) else None
            # parent node
            node_p = find_first_common_ancestor(node_w, node_r) if node_r else None
            node_p_treeposition = node_p.treeposition()
            # child subtree of node_p that includes node_w
            ancestor_w = node_p[node_w.treeposition()[len(node_p_treeposition)]]
            # child subtree of node_p that includes node_r
            ancestor_r = node_p[node_r.treeposition()[len(node_p_treeposition)]]
            node_p_parent = node_p.parent()
            node_p_right_siblings = node_p.right_siblings()
            node_p_right_sibling = node_p_right_siblings[0] if node_p_right_siblings else None

            # now make the list of features
            feats.append(token.lower())
            feats.append(pos_tag)
            feats.append('B-SENT' if token_num == 0 else 'C-SENT')
            feats.extend(parse_node_features([node_p, ancestor_w, ancestor_r, node_p_parent, node_p_right_sibling]))

            feat_lists.append(feats)
            labels.append(label)

    return feat_lists, labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='JSON file from convert_rst_discourse_tb.py')
    parser.add_argument('output_path', help='TSV output file to be used by crf++')
    args = parser.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)

    with open(args.output_path, 'w') as outfile:
        for doc in data:
            feat_lists, labels = extract_segmentation_features(doc)
            for feat_list, label in zip(feat_lists, labels):
                print('\t'.join(feat_list + [label]), file=outfile)

            print(''.join(['\t' for x in range(len(feat_lists[0]) + 1)]), file=outfile)




if __name__ == '__main__':
    main()
