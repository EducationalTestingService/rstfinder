
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
    :param doc_dict: A dictionary of edu_start_indices, tokens, syntax_trees,
                token_tree_positions, and pos_tags for a document, as
                extracted by convert_rst_discourse_tb.py.
    '''
    labels = []
    feat_lists = []
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
        tree = HeadedParentedTree.fromstring(tree_str)
        for token_num, (token, tree_position, pos_tag) \
                in enumerate(zip(sent_tokens, sent_tree_positions, pos_tags)):
            feats = []
            label = 'B-EDU' if (sent_num, token_num) in edu_starts else 'C-EDU'

            # TODO: all of the stuff below needs to be checked

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
            feats.append('B-SENT' if token_num == 0 else 'C-SENT')
            feats.extend(parse_node_features([node_p,
                                              ancestor_w,
                                              ancestor_r,
                                              node_p_parent,
                                              node_p_right_sibling]))

            feat_lists.append(feats)
            labels.append(label)

    return feat_lists, labels

class Segmenter():
    def __init__(self, model_path):
        self.model_path = model_path

    def segment_document(self, doc_dict):
        logging.info('segmenting document...')

        # Extract features.
        # TODO interact with crf++ via cython, etc.?
        tmpfile = NamedTemporaryFile('w')
        feat_lists, _ = extract_segmentation_features(doc_dict)
        for feat_list in feat_lists:
            print('\t'.join(feat_list + ["?"]), file=tmpfile)
        tmpfile.flush()

        # Get predictions from the CRF++ model.
        crf_output = subprocess.check_output(shlex.split(
            'crf_test -m {} {}'.format(self.model_path, tmpfile.name))) \
            .decode('utf-8').strip()
        tmpfile.close()

        # an index into the list of tokens for this document indicating where
        # the current sentence started
        sent_start_index = 0

        # an index into the list of sentences
        sent_num = 0

        edu_number = 0

        # Check that the input is not blank.
        all_tokens = doc_dict['tokens']
        if not all_tokens:
            doc_dict['edu_start_indices'] = []
            return

        # Construct the set of EDU start index tuples (sentence number, token
        # number, EDU number).
        cur_sent = all_tokens[0]
        edu_start_indices = []
        for tok_index, line in enumerate(crf_output.split('\n')):
            if tok_index - sent_start_index >= len(cur_sent):
                sent_start_index += len(cur_sent)
                sent_num += 1
                cur_sent = all_tokens[sent_num] if sent_num < len(
                    all_tokens) else None
            # Start a new EDU where the CRF predicts "B-EDU".
            # Also, force new EDUs to start at the beginnings of sentences to
            # account for the rare cases where the CRF does not predict "B-EDU"
            # at the beginning of a new sentence (CRF++ can only learn this as
            # a soft constraint).
            start_of_sentence = (tok_index - sent_start_index == 0)
            token_label = line.split()[-1]
            if token_label == "B-EDU" or start_of_sentence:
                if start_of_sentence and token_label != "B-EDU":
                    logging.info("The CRF segmentation model did not predict" +
                                 " B-EDU at the start of a sentence. A new" +
                                 " EDU will be started regardless, to ensure." +
                                 " consistency with the RST annotations.")

                edu_start_indices.append(
                    (sent_num, tok_index - sent_start_index, edu_number))
                edu_number += 1

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
            raise ValueError('An EDU ({}) crosses sentences: (sent {}, tok {}) => (sent {}, tok {})'
                             .format(prev_edu_index,
                                     prev_sent_index, prev_tok_index,
                                     sent_index, tok_index))
    return res


def extract_tagged_doc_edus(doc_dict):
    edu_start_indices = doc_dict['edu_start_indices']
    res = [list(zip(edu_words, edu_tags))
           for edu_words, edu_tags
           in zip(extract_edus_tokens(edu_start_indices, doc_dict['tokens']),
                  extract_edus_tokens(edu_start_indices, doc_dict['pos_tags']))]
    return res
