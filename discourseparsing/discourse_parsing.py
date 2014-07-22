
'''
License
-------
Copyright (c) 2014, Educational Testing Service and Kenji Sagae
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Description
-----------
This is a python version of a shift-reduce RST discourse parser,
originally written by Kenji Sagae in perl.

'''

import os
import re
import logging
from collections import namedtuple, Counter
from operator import itemgetter

from nltk.tree import Tree, ParentedTree
import numpy as np
import skll

from discourseparsing.tree_util import (collapse_binarized_nodes,
                                        HeadedParentedTree)
from discourseparsing.discourse_segmentation import extract_tagged_doc_edus


ShiftReduceAction = namedtuple("ShiftReduceAction", ["type", "label"])
ScoredAction = namedtuple("ScoredAction", ["action", "score"])
logger = logging.getLogger(__name__)


class Parser(object):
    leftwall_w = 'LEFTWALL'
    leftwall_p = 'LEFTWALL'
    rightwall_w = 'RIGHTWALL'
    rightwall_p = 'RIGHTWALL'

    def __init__(self, max_acts, max_states, n_best):
        self.max_acts = max_acts
        self.max_states = max_states
        self.n_best = n_best
        self.model = None
        self.model_action_list = None

    def load_model(self, model_path):
        self.model = skll.learner.Learner.from_file(
            os.path.join(model_path,
                         'rst_parsing_all_feats_LogisticRegression.model'))

    def _get_model_actions(self):
        '''
        This creates a list of ShiftReduceAction objects for the list of
        classifier labels.  This is used later when parsing, to decide which
        action to take based on a list of scores.
        '''
        if self.model_action_list is None:
            self.model_action_list = []
            for x in self.model.label_list:
                act = ShiftReduceAction(type=x[0], label=x[2:])
                self.model_action_list.append(act)
        return self.model_action_list

    @staticmethod
    def _add_word_and_pos_feats(feats, prefix, words, pos_tags):
        '''
        This is for adding word and POS features for the head EDU of a subtree.
        It also adds specially marked features for the first 2 and last 1 token.
        `feats` is the existing list of features.
        The prefix indicates where the tokens are from (S0, S1, S2, Q0, Q1).
        '''

        # Do not add any word or POS features for the LEFTWALL or RIGHTWALL.
        # That information should be available in the nonterminal features.
        if pos_tags == [Parser.leftwall_p] or pos_tags == [Parser.rightwall_p]:
            assert words == [Parser.leftwall_w] or words == [Parser.rightwall_w]
            return

        # first 2 and last 1
        feats.append('{}w:{}:::0'.format(prefix, words[0]))
        feats.append('{}p:{}:::0'.format(prefix, pos_tags[0]))
        feats.append('{}w:{}:::-1'.format(prefix, words[-1]))
        feats.append('{}p:{}:::-1'.format(prefix, pos_tags[-1]))
        feats.append('{}w:{}:::1'.format(prefix, (words[1]
                                                  if len(words) > 1
                                                  else "")))
        feats.append('{}p:{}:::1'.format(prefix, (pos_tags[1]
                                                  if len(pos_tags) > 1
                                                  else "")))

        # first bigram
        # feats.append('{}w2:{}:{}'.format(prefix,
        #                                  words[0], (words[1]
        #                                             if len(words) > 1
        #                                             else "")))
        # feats.append('{}p2:{}:{}'.format(prefix,
        #                                  words[0], (pos_tags[1]
        #                                             if len(pos_tags) > 1
        #                                             else "")))

        for word in words:
            feats.append("{}w:{}".format(prefix, word))
        for pos_tag in pos_tags:
            feats.append("{}p:{}".format(prefix, pos_tag))

    @staticmethod
    def _find_edu_head_node(rst_node, doc_dict):
        '''
        Find the EDU head node, which is the node whose head is
        "the word with the highest occurrence as a lexical head"
        (Soricut & Marco, 2003, Sec 4.1).

        There can be ties, which the paper doesn't mention.
        This code just finds the leftmost, using np.argmin on tree depths.
        '''

        # return None for the left wall
        head_idx = rst_node["head_idx"]
        if head_idx == -1:
            return None

        head_words = rst_node["head"]

        edu_start_indices = doc_dict['edu_start_indices'][head_idx]
        tree_idx, start_tok_idx, _ = edu_start_indices
        tree = doc_dict['syntax_trees_objs'][tree_idx]
        end_tok_idx = start_tok_idx + len(head_words)
        preterminals = [x for x in tree.subtrees()
                        if isinstance(x[0], str)][start_tok_idx:end_tok_idx]
        # filter out punctuation
        maximal_nodes = [node.find_maximal_head_node() for node in preterminals
                         if re.search(r'[A-Za-z]', node.label())]
        if len(maximal_nodes) == 0:
            logging.warning("EDU head only contained punctuation: {}"
                            .format(preterminals))
            return None
        depths = [len(node.treeposition()) for node in maximal_nodes]
        mindepth_idx = np.argmin(depths)
        res = maximal_nodes[mindepth_idx]

        return res

    @staticmethod
    def mkfeats(prevact, sent, stack, doc_dict):
        '''
        get features of the parser state represented
        by the current stack and queue
        '''

        feats = []

        # initialize some local variables for top stack and next queue items
        s0 = stack[-1]
        s1 = {"nt": "TOP", "head": [Parser.leftwall_w], "hpos": [Parser.leftwall_p],
              "tree": [], "start_idx": -1, "end_idx": -1, "head_idx": -1}
        s2 = {"nt": "TOP", "head": [Parser.leftwall_w], "hpos": [Parser.leftwall_p],
              "tree": [], "start_idx": -1, "end_idx": -1, "head_idx": -1}
        s3 = {"nt": "TOP", "head": [Parser.leftwall_w], "hpos": [Parser.leftwall_p],
              "tree": [], "start_idx": -1, "end_idx": -1, "head_idx": -1}
        stack_len = len(stack)
        if stack_len > 1:
            s1 = stack[stack_len - 2]
        if stack_len > 2:
            s2 = stack[stack_len - 3]
        if stack_len > 3:
            s3 = stack[stack_len - 4]

        q0w = [Parser.rightwall_w]
        q0p = [Parser.rightwall_p]
        q1w = [Parser.rightwall_w]
        q1p = [Parser.rightwall_p]
        if len(sent) > 0:
            q0w = sent[0]["head"]
            q0p = sent[0]["hpos"]
        if len(sent) > 1:
            q1w = sent[1]["head"]
            q1p = sent[1]["hpos"]

        # previous action feature
        feats.append("PREV:{}:{}".format(prevact.type, prevact.label))

        # stack nonterminal symbol features
        feats.append("S0nt:{}".format(s0["nt"]))
        if len(s0["tree"]) > 0 and isinstance(s0["tree"][0], Tree):
            feats.append("S0lnt:{}".format(s0["tree"][0].label()))
            feats.append("S0rnt:{}".format(s0["tree"][-1].label()))

        feats.append("S1nt:{}".format(s1["nt"]))
        if len(s1["tree"]) > 0  and isinstance(s1["tree"][0], Tree):
            feats.append("S1lnt:{}".format(s1["tree"][0].label()))
            feats.append("S1rnt:{}".format(s1["tree"][-1].label()))

        feats.append("S2nt:{}".format(s2["nt"]))
        feats.append("S3nt:{}".format(s3["nt"]))

        feats.append("S0nt:{}^S1nt:{}".format(s0["nt"], s1["nt"]))

        # features for the words and POS tags of the heads of the first and
        # last tokens of the heads of the top stack and next input queue items
        Parser._add_word_and_pos_feats(feats, 'S0', s0['head'], s0['hpos'])
        Parser._add_word_and_pos_feats(feats, 'S1', s1['head'], s1['hpos'])
        Parser._add_word_and_pos_feats(feats, 'S2', s2['head'], s2['hpos'])
        Parser._add_word_and_pos_feats(feats, 'Q0', q0w, q0p)
        Parser._add_word_and_pos_feats(feats, 'Q1', q1w, q1p)

        # EDU head distance feature
        # (this is in EDUs, not tokens, and -1 is for the left wall)
        dist = s0.get("head_idx") - s1.get("head_idx")
        feats.append("dist:{}".format(dist))

        # whether the EDUS are in the same sentence
        # (edu_start_indices is a list of (sentence #, token #, EDU #) tuples.
        # Also, EDUs don't cross sentence boundaries.)
        start_indices = doc_dict['edu_start_indices']
        s0_start_idx = s0["start_idx"]
        s1_end_idx = s1["end_idx"]
        if s0_start_idx > -1 and s1_end_idx > -1 and \
                start_indices[s0_start_idx][0] == start_indices[s1_end_idx][0]:
            feats.append("S0S1_same_sentence")

        # features of EDU heads
        head_node_s0 = Parser._find_edu_head_node(s0, doc_dict)
        head_node_s1 = Parser._find_edu_head_node(s1, doc_dict)
        head_node_q0 = Parser._find_edu_head_node(sent[0], doc_dict) \
            if sent else None
        if head_node_s0:
            feats.append('S0headnt:{}'.format(head_node_s0.label()))
            feats.append('S0headw:{}'.format(head_node_s0.head_word()))
            feats.append('S0headp:{}'.format(head_node_s0.head_pos()))
            #logging.info('\nHEAD EDU: {}\nFEATS: {}\nTREE: {}\n'.format(' '.join(s0['head']), str(feats[-3:]), doc_dict['syntax_trees'][doc_dict['edu_start_indices'][s0['head_idx']][0]] if s0['head_idx'] > -1 else ""))
        if head_node_s1:
            feats.append('S1headnt:{}'.format(head_node_s1.label()))
            feats.append('S1headw:{}'.format(head_node_s1.head_word()))
            feats.append('S1headp:{}'.format(head_node_s1.head_pos()))
        if head_node_q0:
            feats.append('Q0headnt:{}'.format(head_node_q0.label()))
            feats.append('Q0headw:{}'.format(head_node_q0.head_word()))
            feats.append('Q0headp:{}'.format(head_node_q0.head_pos()))

        # TODO features for the head words of the EDUS

        # TODO parse tree nonterminal features?

        # combinations of features with the previous action
        # for i in range(len(feats)):
        #     feat = feats[i]
        #     # Do not include duplicates.
        #     if not feat.startswith('PREV:'):
        #         feats.append("combo:{}^PREV:{}:{}"
        #                      .format(feats[i], prevact.type, prevact.label))
        return feats

    @staticmethod
    def is_valid_action(act, ucnt, sent, stack):
        if act.type == "U":
            # Do not allow too many consecutive unary reduce actions.
            if ucnt > 2:
                return False

            # Do not allow a reduce action if the stack is empty.
            # (i.e., contains only the leftwall)
            if stack[-1]["head"] == Parser.leftwall_w:
                return False

            # Do not allow unary reduces on internal nodes for binarized rules.
            if stack[-1]["nt"].endswith('*'):
                return False

        # Do not allow shift if there is nothing left to shift.
        if act.type == "S" and not sent:
            return False

        # Do not allow a binary reduce unless there are at least two items in
        # the stack to be reduced (plus the leftwall),
        # with one of them being a nucleus or a partial subtree containing
        # a nucleus, as indicated by a * suffix).
        if act.type == "B":
            # Do not allow B:ROOT unless we will have a complete parse.
            if act.label == "ROOT" and (len(stack) != 2 or sent):
                return False

            # Make sure there are enough items to reduce
            # (including the left wall).
            if act.label != "ROOT" and len(stack) < 3:
                return False

            # Make sure there is a head.
            lc_label = stack[-2]["nt"]
            rc_label = stack[-1]["nt"]
            if not (lc_label.startswith('nucleus')
                    or rc_label.startswith('nucleus')
                    or lc_label.endswith('*')
                    or rc_label.endswith('*')):
                return False

            # Check that partial node labels (ending with *) match the action.
            if lc_label.endswith('*') \
                    and act.label != lc_label and act.label != lc_label[:-1]:
                return False
            if rc_label.endswith('*') \
                    and act.label != rc_label and act.label != rc_label[:-1]:
                return False

        # Default: the action is valid.
        return True

    @staticmethod
    def process_action(act, sent, stack):
        # The R action reduces the stack, creating a non-terminal node
        # with a lexical head coming from the left child
        # (this is a confusing name, but it refers to the direction of
        # the dependency arrow).
        if act.type == "B":
            tmp_rc = stack.pop()
            tmp_lc = stack.pop()
            new_tree = Tree("({})".format(act.label))
            new_tree.append(tmp_lc["tree"])
            new_tree.append(tmp_rc["tree"])

            # Reduce right, making the left node the head
            # because it is the nucleus (or a partial tree containing the
            # nucleus, indicated by a * suffix) or the leftwall.
            if tmp_lc["nt"].startswith('nucleus:') \
                    or tmp_lc["nt"].endswith('*') \
                    or (act.type == 'B' and act.label == 'ROOT'):
                tmp_item = {"head_idx": tmp_lc["head_idx"],
                            "start_idx": tmp_lc["start_idx"],
                            "end_idx": tmp_rc["end_idx"],
                            "nt": act.label,
                            "tree": new_tree,
                            "head": tmp_lc["head"],
                            "hpos": tmp_lc["hpos"]}
            # Reduce left, making the right node the head
            # because it is the nucleus (or a partial tree containing the
            # nucleus, indicated by a * suffix)
            elif tmp_rc["nt"].startswith('nucleus:') \
                    or tmp_rc["nt"].endswith('*'):
                tmp_item = {"head_idx": tmp_rc["head_idx"],
                            "start_idx": tmp_lc["start_idx"],
                            "end_idx": tmp_rc["end_idx"],
                            "nt": act.label,
                            "tree": new_tree,
                            "head": tmp_rc["head"],
                            "hpos": tmp_rc["hpos"]}
            else:
                raise ValueError("Unexpected binary reduce.\n" +
                                 "act = {}:{}\n tmp_lc = {}\ntmp_rc = {}"
                                 .format(act.type, act.label, tmp_lc, tmp_rc))

            stack.append(tmp_item)

        # The U action creates a unary chain (e.g., "(NP (NP ...))").
        if act.type == "U":
            tmp_c = stack.pop()
            new_tree = Tree("({})".format(act.label))
            new_tree.append(tmp_c["tree"])
            tmp_item = {"head_idx": tmp_c["head_idx"],
                        "start_idx": tmp_c["start_idx"],
                        "end_idx": tmp_c["end_idx"],
                        "nt": act.label,
                        "tree": new_tree,
                        "head": tmp_c["head"],
                        "hpos": tmp_c["hpos"]}
            stack.append(tmp_item)

        # The S action gets the next input token
        # and puts it on the stack.
        if act.type == "S":
            stack.append(sent.pop(0))

    @staticmethod
    def initialize_edu_data(edus):
        '''
        Create a representation of the list of EDUS that make up the input.
        '''

        wnum = 0  # counter for distance features
        res = []
        for edu_index, edu in enumerate(edus):
            # lowercase all words
            edu_words = [x[0].lower() for x in edu]
            edu_pos_tags = [x[1] for x in edu]

            # make a dictionary for each EDU
            new_tree = Tree('(text)')
            new_tree.append('{}'.format(edu_index))
            tmp_item = {"head_idx": wnum,
                        "start_idx": wnum,
                        "end_idx": wnum,
                        "nt": "text",
                        "head": edu_words,
                        "hpos": edu_pos_tags,
                        "tree": new_tree}
            wnum += 1
            res.append(tmp_item)
        return res

    def parse(self, doc_dict, gold_actions=None, make_features=True):
        '''
        `doc_dict` is a dictionary with EDU segments, parse trees, etc.
        See `convert_rst_discourse_tb.py`.

        If `gold_actions` is specified, then the parser will behave as if in
        training mode.

        If `make_features` and `gold_actions` are specified, then the parser
        will yield (action, features) tuples instead of trees
        (e.g., to produce training examples).
        This will have no effect if `gold_actions` is not provided.
        Disabling `make features` can be useful for debugging and testing.
        '''

        logging.info('RST parsing document...')

        states = []
        completetrees = []
        tagged_edus = extract_tagged_doc_edus(doc_dict)

        sent = self.initialize_edu_data(tagged_edus)

        # precompute syntax tree objects so this only needs to be done once
        if 'syntax_trees_objs' not in doc_dict \
                or len(doc_dict['syntax_trees_objs']) \
                != len(doc_dict['syntax_trees']):
            doc_dict['syntax_trees_objs'] = []
            for tree_str in doc_dict['syntax_trees']:
                doc_dict['syntax_trees_objs'].append(
                    HeadedParentedTree(tree_str))

        # initialize the stack
        stack = []

        tmp_item = {"head_idx": -1,
                    "start_idx": -1,
                    "end_idx": -1,
                    "nt": Parser.leftwall_w,
                    "tree": Tree("({})".format(Parser.leftwall_w)),
                    "head": [Parser.leftwall_w],
                    "hpos": [Parser.leftwall_p]}
        stack.append(tmp_item)

        prevact = ShiftReduceAction(type="S", label="text")
        ucnt = 0  # number of consecutive unary reduce actions

        # insert an initial state on the state list
        tmp_state = {"prevact": prevact,
                     "ucnt": 0,
                     "score": 0.0,  # log probability
                     "nsteps": 0,
                     "stack": stack,
                     "sent": sent}
        states.append(tmp_state)

        # loop while there are states to process
        while states:
            states.sort(key=itemgetter("score"), reverse=True)
            states = states[:self.max_states]

            cur_state = states.pop(0)  # should maybe replace this with a deque
            logging.debug("cur_state prevact: {}:{}, score: {}, num. states: {}"
                          .format(cur_state["prevact"].type,
                                  cur_state["prevact"].label,
                                  cur_state["score"],
                                  len(states)))

            # check if the current state corresponds to a complete tree
            if len(cur_state["sent"]) == 0 and len(cur_state["stack"]) == 1:
                tree = cur_state["stack"][0]["tree"]

                # remove the dummy LEFTWALL node
                assert tree[0].label() == Parser.leftwall_p
                del tree[0]

                # collapse binary branching * rules in the output
                output_tree = ParentedTree(tree.pprint())
                collapse_binarized_nodes(output_tree)

                completetrees.append({"tree": output_tree,
                                      "score": cur_state["score"]})
                logging.debug('complete tree found')

                # stop if we have found enough trees
                if gold_actions is not None or (len(completetrees) >=
                                                self.n_best):
                    break

                # otherwise, move on to the next best state
                continue

            stack = cur_state["stack"]
            sent = cur_state["sent"]
            prevact = cur_state["prevact"]
            ucnt = cur_state["ucnt"]

            # extract features
            feats = self.mkfeats(prevact, sent, stack, doc_dict)

            # Compute the possible actions given this state.
            # During training, print them out.
            # During parsing, score them according to the model and sort.
            scored_acts = []
            if gold_actions is not None:
                # take the next action from gold_actions
                act = gold_actions.pop(0) if gold_actions else None
                if act is None:
                    logger.error('Ran out of gold actions for state %s and ' +
                                 'gold_actions %s', cur_state, gold_actions)
                    break

                assert act.type != 'S' or act.label == "text"

                if make_features:
                    if not (act == cur_state["prevact"] and act.type == 'U'):
                        yield ('{}:{}'.format(act.type, act.label), feats)

                scored_acts.append(ScoredAction(act, 0.0))  # logprob
            else:
                vectorizer = self.model.feat_vectorizer
                examples = skll.data.ExamplesTuple(None, None,
                                                   vectorizer.transform(Counter(feats)),
                                                   vectorizer)
                scores = [np.log(x) for x in self.model.predict(examples)[0]]

                # Convert the string labels from the classifier back into
                # ShiftReduceAction objects and sort them by their scores
                scored_acts = sorted(zip(self._get_model_actions(),
                                         scores),
                                     key=itemgetter(1),
                                     reverse=True)
                #print('\n'.join(['{} {:.4g}'.format(x.action, x.score) for x in scored_acts]), file=sys.stderr)
                #print('\n', file=sys.stderr)

            # If parsing, verify the validity of the actions.
            if gold_actions is None:
                scored_acts = [x for x in scored_acts
                               if self.is_valid_action(x[0], ucnt, sent, stack)]
            else:
                for x in scored_acts:
                    assert self.is_valid_action(x[0], ucnt, sent, stack)

            # Don't exceed the maximum number of actions
            # to consider for a parser state.
            scored_acts = scored_acts[:self.max_acts]

            while scored_acts:
                if self.max_acts > 1:
                    # Make copies of the input queue and stack.
                    # This is not necessary if we are doing greedy parsing.
                    # Note that we do not need to make deep copies because
                    # the reduce actions do not modify the subtrees.  They
                    # only create new trees that have them as children.
                    # This ends up making something like a parse forest.
                    sent = list(cur_state["sent"])
                    stack = list(cur_state["stack"])
                prevact = cur_state["prevact"]
                ucnt = cur_state["ucnt"]

                action, score = scored_acts.pop(0)

                # If the action is a unary reduce, increment the count.
                # Otherwise, reset it.
                ucnt = ucnt + 1 if action.type == "U" else 0

                self.process_action(action, sent, stack)

                # Add the newly created state
                tmp_state = {"prevact": action,
                             "ucnt": ucnt,
                             "score": cur_state["score"] + score,
                             "nsteps": cur_state["nsteps"] + 1,
                             "stack": stack,
                             "sent": sent}
                states.append(tmp_state)

        if not completetrees:
            # Default to a flat tree if there is no complete parse.
            new_tree = Tree("(ROOT)")
            for i in range(len(tagged_edus)):
                tmp_child = Tree('(text)')
                tmp_child.append(i)
                new_tree.append(tmp_child)
            completetrees.append({"tree": new_tree, "score": 0.0})

        if gold_actions is None or not make_features:
            for t in completetrees:
                yield t
