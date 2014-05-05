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


class HeadedParentedTree(ParentedTree):
    '''
    A subclass of nltk.tree.ParentedTree
    that also returns heads using head rules from Michael Collins's
    1999 thesis, Appendix A.  See the head() function.
    '''

    start_points = {"ADJP": "L",
                    "ADVP": "R",
                    "CONJP": "R",
                    "FRAG": "R",
                    "INTJ": "L",
                    "LST": "R",
                    "NAC": "L",
                    "PP": "R",
                    "PRN": "L",
                    "PRT": "R",
                    "QP": "L",
                    "RRC": "R",
                    "S": "L",
                    "SBAR": "L",
                    "SBARQ": "L",
                    "SINV": "L",
                    "SQ": "L",
                    "UCP": "R",
                    "VP": "L",
                    "WHADJP": "L",
                    "WHADVP": "R",
                    "WHNP": "L",
                    "WHPP": "R",
                    "NX": "L",
                    "X": "L"}

    priority_list = {"ADJP": ["NNS", "QP", "NN", "$", "ADVP", "JJ", "VBN",
                              "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW",
                              "RBR", "RBS", "SBAR", "RB"],
                     "ADVP": ["RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD",
                              "JJR", "JJ", "IN", "NP", "JJS", "NN"],
                     "CONJP": ["CC", "RB", "IN"],
                     "FRAG": [],
                     "INTJ": [],
                     "LST": ["LS", ":"],
                     "NAC": ["NN", "NNS", "NNP", "NNPS", "NP", "NAC", "EX",
                             "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS",
                             "JJR", "ADJP", "FW"],
                     "PP": ["IN", "TO", "VBG", "VBN", "RP", "FW"],
                     "PRN": [],
                     "PRT": ["RP"],
                     "QP": ["$", "IN", "NNS", "NN", "JJ", "RB", "DT", "CD",
                            "NCD", "QP", "JJR", "JJS"],
                     "RRC": ["VP", "NP", "ADVP", "ADJP", "PP"],
                     "S": ["TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"],
                     "SBAR": ["WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT",
                              "S", "SQ", "SINV", "SBAR", "FRAG"],
                     "SBARQ": ["SQ", "S", "SINV", "SBARQ", "FRAG"],
                     "SINV": ["VBZ", "VBD", "VBP", "VB", "MD", "VP", "S",
                              "SINV", "ADJP", "NP"],
                     "SQ": ["VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"],
                     "UCP": [],
                     "VP": ["TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG",
                            "VBP", "VP", "ADJP", "NN", "NNS", "NP"],
                     "WHADJP": ["CC", "WRB", "JJ", "ADJP"],
                     "WHADVP": ["CC", "WRB"],
                     "WHNP": ["WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"],
                     "WHPP": ["IN", "TO", "FW"],
                     "NX": [],
                     "X": []}

    def __init__(self, node_or_str, children=None):
        self._head = None
        super(HeadedParentedTree, self).__init__(node_or_str, children)

    def _search_children(self, search_list, start_point):
        '''
        A helper function for finding heads of noun phrases.
        This finds the first node whose label is in search_list, starting
        from start_point, either "L" for left (i.e., 0) or "R" for right.
        '''

        assert start_point == "L" or start_point == "R"

        head_index = None
        num_children = len(self)
        children = list(self)

        # reverse the list if we start from the right
        if start_point == "R":
            children.reverse()

        for i, child in enumerate(children):
            if child.label() in search_list:
                head_index = i
                break

        # correct the index if we reversed the list to start from the right
        if start_point == "R" and head_index is not None:
            head_index = num_children - 1 - head_index

        return head_index

    def head(self):
        '''
        Head finding rules, following Michael Collins' head rules
        (from his 1999 Ph.D. thesis, Appendix A).

        A default of the leftmost child was added for NX nodes, which aren't
        discussed in Collin's thesis.  This follows the Stanford Rarser
        (http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/CollinsHeadFinder.html).
        '''
        if self._head is None:
            num_children = len(self)
            head_index = None

            if num_children < 2:
                # shortcut for when there is only one child
                self._head = self[0]
                return self._head

            # special case: NPs
            if self.label() == 'NP':
                # If last node is POS, that's the head
                if self[-1].label() == "POS":
                    head_index = num_children - 1

                # Otherwise, look right to left for NN, NNP, NNPS, NNS, NX, POS, or JJR.
                if head_index is None:
                    head_index = self._search_children(["NN", "NNP", "NNPS", "NNS", "NX", "POS", "JJR"],
                                                       "R")

                # Otherwise, search left to right for NP.
                if head_index is None:
                    head_index = self._search_children(["NP"],
                                                       "L")

                # Otherwise, search right to left for $, ADJP, PRN.
                if head_index is None:
                    head_index = self._search_children(["$", "ADJP", "PRN"],
                                                       "R")

                # Otherwise, search right to left for CD.
                if head_index is None:
                    head_index = self._search_children(["CD"],
                                                       "R")

                # Otherwise, search right to left for JJ, JJS, RB, or QP.
                if head_index is None:
                    head_index = self._search_children(["JJ", "JJS", "RB", "QP"],
                                                       "R")

                # Otherwise, return the last child.
                if head_index is None:
                    head_index = num_children - 1

            else: # typical cases
                start_point = self.start_points[self.label()]

                # Try looking for each symbol in the priority list.
                # Stop at the first match.
                for symbol in self.priority_list[self.label()]:
                    head_index = self._search_children([symbol], start_point)
                    if head_index is not None:
                        break

                if head_index is None:
                    # If none of the symbols given in the priority list
                    # for this label was found, then default to the first
                    # child from the left or right, as specified by the
                    # starting points table.
                    head_index = 0 if start_point == 'L' else num_children - 1

            # special case: coordination.
            # After finding the head, check to see if its left sibling is a
            # conjunction.  If so, move the head index left 2.
            if 'CC' in {x.label() for x in self}:
                if head_index > 2 and self[head_index - 1].label() == 'CC':
                    head_index -= 2

            # cache the result
            self._head = self[head_index]

        return self._head

    def head_preterminal(self):
        res = self
        while not isinstance(res[0], str):
            res = res.head()
        return res

    def head_word(self):
        return self.head_preterminal()[0]

    def head_pos(self):
        return self.head_preterminal().label()


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


def parse_node_features(nodes):
    for node in nodes:
        if node is None:
            yield ''
            yield ''
            continue

        node_head_preterminal = node.head_preterminal()
        yield '{}({})'.format(node.label(), node_head_preterminal[0].lower()) if node else ''
        yield '{}({})'.format(node.label(), node_head_preterminal.label()) if node else ''


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
        tree = HeadedParentedTree(tree_str)
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

            node_p, ancestor_w, ancestor_r, node_p_parent, node_p_right_sibling = None, None, None, None, None
            if node_r:
                node_p = find_first_common_ancestor(node_w, node_r)
                node_p_treeposition = node_p.treeposition()
                # child subtree of node_p that includes node_w
                ancestor_w = node_p[node_w.treeposition()[len(node_p_treeposition)]]
                # child subtree of node_p that includes node_r
                ancestor_r = node_p[node_r.treeposition()[len(node_p_treeposition)]]
                node_p_parent = node_p.parent()
                node_p_right_sibling = node_p.right_sibling()


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
