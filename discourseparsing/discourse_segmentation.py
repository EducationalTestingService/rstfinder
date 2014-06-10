
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
        tree = HeadedParentedTree(tree_str)
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
