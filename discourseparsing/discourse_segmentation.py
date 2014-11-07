# License: MIT

from tempfile import NamedTemporaryFile
import shlex
import subprocess
import logging

from discourseparsing.tree_util import (HeadedParentedTree,
                                        find_first_common_ancestor)


def parse_node_features(nodes):
    for node in nodes:
        if node is None:
            yield '*NULL*'
            yield '*NULL*'
            continue

        node_head_preterminal = node.head_preterminal()
        yield ('{}({})'.format(node.label(),
                               node_head_preterminal[0].lower())
               if node else '*NULL*')
        yield ('{}({})'.format(node.label(),
                               node_head_preterminal.label())
               if node else '*NULL*')


def extract_segmentation_features(doc_dict):
    '''
    This extracts features for use in the discourse segmentation CRF. Note that
    the CRF++ template makes it so that the features for the current word and
    2 previous and 2 next words are used for each word.

    :param doc_dict: A dictionary of edu_start_indices, tokens, syntax_trees,
                     token_tree_positions, and pos_tags for a document, as
                     extracted by convert_rst_discourse_tb.py.
    :returns: a list of lists of lists of features (one feature list per word
              per sentence), and a list of lists of labels (one label per word
              per sentence)
    '''

    labels_doc = []
    feat_lists_doc = []

    if 'edu_start_indices' in doc_dict:
        edu_starts = {(x[0], x[1]) for x in doc_dict['edu_start_indices']}
    else:
        # if none available, just say the whole document is one EDU
        edu_starts = {(0, 0)}

    for sent_num, (sent_tokens, tree_str, sent_tree_positions, pos_tags) \
            in enumerate(zip(doc_dict['tokens'],
                             doc_dict['syntax_trees'],
                             doc_dict['token_tree_positions'],
                             doc_dict['pos_tags'])):

        labels_sent = []
        feat_lists_sent = []

        tree = HeadedParentedTree.fromstring(tree_str)
        for token_num, (token, tree_position, pos_tag) \
                in enumerate(zip(sent_tokens, sent_tree_positions, pos_tags)):
            feats = []
            label = 'B-EDU' if (sent_num, token_num) in edu_starts else 'C-EDU'

            # POS tags and words for lexicalized parse nodes
            # from 3.2 of Bach et al., 2012.
            # preterminal node for the current word
            node_w = tree[tree_position]
            # node for the word to the right
            node_r = tree[sent_tree_positions[token_num + 1]] if token_num + \
                1 < len(sent_tree_positions) else None
            # parent node

            node_p, ancestor_w, ancestor_r = None, None, None
            node_p_parent, node_p_right_sibling = None, None
            if node_r:
                node_p = find_first_common_ancestor(node_w, node_r)
                node_p_treeposition = node_p.treeposition()
                node_p_len = len(node_p_treeposition)
                # child subtree of node_p that includes node_w
                ancestor_w = node_p[node_w.treeposition()[node_p_len]]
                # child subtree of node_p that includes node_r
                ancestor_r = node_p[node_r.treeposition()[node_p_len]]
                node_p_parent = node_p.parent()
                node_p_right_sibling = node_p.right_sibling()

            # now make the list of features
            feats.append(token.lower())
            feats.append(pos_tag)
            feats.extend(parse_node_features([node_p,
                                              ancestor_w,
                                              ancestor_r,
                                              node_p_parent,
                                              node_p_right_sibling]))

            feat_lists_sent.append(feats)
            labels_sent.append(label)
        feat_lists_doc.append(feat_lists_sent)
        labels_doc.append(labels_sent)

    return feat_lists_doc, labels_doc


class Segmenter():
    def __init__(self, model_path):
        self.model_path = model_path

    def segment_document(self, doc_dict):
        doc_id = doc_dict["doc_id"]
        logging.info('segmenting document, doc_id = {}'.format(doc_id))

        # Extract features.
        tmpfile = NamedTemporaryFile('w')
        feat_lists_doc, _ = extract_segmentation_features(doc_dict)
        for feat_lists_sent in feat_lists_doc:
            for feat_list_word in feat_lists_sent:
                print('\t'.join(feat_list_word + ["?"]), file=tmpfile)
            print('\n', file=tmpfile)
        tmpfile.flush()

        # Get predictions from the CRF++ model.
        # TODO interact with crf++ via cython, etc.?
        crf_output = subprocess.check_output(shlex.split(
            'crf_test -m {} {}'.format(self.model_path, tmpfile.name))) \
            .decode('utf-8').strip()
        tmpfile.close()

        # an index into the list of sentences
        sent_num = 0
        edu_num = 0

        # Check that the input is not blank.
        all_tokens = doc_dict['tokens']
        if not all_tokens:
            doc_dict['edu_start_indices'] = []
            return

        # Construct the set of EDU start index tuples (sentence number, token
        # number, EDU number).
        edu_start_indices = []

        for sent_num, crf_output_sent in enumerate(crf_output.split('\n\n')):
            for tok_num, line in enumerate(crf_output_sent.split('\n')):
                # Start a new EDU where the CRF predicts "B-EDU" and
                # at the beginnings of sentences.
                token_label = line.split()[-1]
                if token_label == "B-EDU" or tok_num == 0:
                    edu_start_indices.append((sent_num, tok_num, edu_num))
                    edu_num += 1

        # Check that all sentences are covered by the output list of EDUs,
        # and that every new sentence starts an EDU.
        assert set(range(len(doc_dict['tokens']))) \
            == {x[0] for x in edu_start_indices if x[1] == 0}

        doc_dict['edu_start_indices'] = edu_start_indices


def extract_edus_tokens(edu_start_indices, tokens_doc):
    res = []

    # check for blank input.
    if not edu_start_indices:
        return res

    # add a dummy index pair representing the end of the document
    tmp_indices = edu_start_indices + [[edu_start_indices[-1][0] + 1,
                                        0,
                                        edu_start_indices[-1][2] + 1]]

    for (prev_sent_index, prev_tok_index, prev_edu_index), \
            (sent_index, tok_index, _) \
            in zip(tmp_indices, tmp_indices[1:]):
        if sent_index == prev_sent_index and tok_index > prev_tok_index:
            res.append(tokens_doc[prev_sent_index][prev_tok_index:tok_index])
        elif sent_index > prev_sent_index and tok_index == 0:
            res.append(tokens_doc[prev_sent_index][prev_tok_index:])
        else:
            raise ValueError(("An EDU ({}) crosses sentences: " +
                              "(sent {}, tok {}) => (sent {}, tok {})")
                             .format(prev_edu_index, prev_sent_index,
                                     prev_tok_index, sent_index, tok_index))
    return res


def extract_tagged_doc_edus(doc_dict):
    edu_start_indices = doc_dict['edu_start_indices']
    res = [list(zip(edu_words, edu_tags))
           for edu_words, edu_tags
           in zip(extract_edus_tokens(edu_start_indices, doc_dict['tokens']),
                  extract_edus_tokens(edu_start_indices,
                                      doc_dict['pos_tags']))]
    return res
