
'''
License
-------
Copyright (c) 2014, Kenji Sagae
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
import logging
import re
from collections import namedtuple, Counter
from operator import itemgetter
from copy import deepcopy

from nltk.tree import ParentedTree
import numpy as np
import skll

from discourseparsing.tree_util import collapse_binarized_nodes


ScoredAction = namedtuple('ScoredAction', ['action', 'score'])
logger = logging.getLogger(__name__)


class Parser(object):

    def __init__(self, max_acts, max_states, n_best):
        self.max_acts = max_acts
        self.max_states = max_states
        self.n_best = n_best
        self.model = None

    def load_model(self, model_path):
        self.model = skll.learner.Learner.from_file(
            os.path.join(model_path,
                         'rst_parsing_all_feats_LogisticRegression.model'))

    @staticmethod
    def mkfeats(prevact, sent, stack):
        '''
        get features of the parser state represented
        by the current stack and queue
        '''

        nw1 = ["RightWall"]
        nw2 = ["RightWall"]
        # nw3 = ["RightWall"]

        np1 = ["RW"]
        np2 = ["RW"]
        # np3 = ["RW"]

        s0 = stack[-1]
        s1 = {'nt': "TOP", 'head': ["LeftWall"], 'hpos': ["LW"], 'tree': []}
        s2 = {'nt': "TOP", 'head': ["LeftWall"], 'hpos': ["LW"], 'tree': []}
        s3 = {'nt': "TOP", 'head': ["LeftWall"], 'hpos': ["LW"], 'tree': []}

        if len(sent) > 0:
            nw1 = sent[0]['head']
            np1 = sent[0]['hpos']
        if len(sent) > 1:
            nw2 = sent[1]['head']
            np2 = sent[1]['hpos']
        # if len(sent) > 2:
        #     nw3 = sent[2]['head']
        #     np3 = sent[2]['hpos']

        stack_len = len(stack)
        if stack_len > 1:
            s1 = stack[stack_len - 2]
        if stack_len > 2:
            s2 = stack[stack_len - 3]
        if stack_len > 3:
            s3 = stack[stack_len - 4]

        feats = []

        feats.append("PREV:{}".format(prevact))

        # features of the 0th item on the stack
        for word in s0['head']:
            feats.append("S0w:{}".format(word))
        for pos_tag in s0['hpos']:
            feats.append("S0p:{}".format(pos_tag))
        feats.append("S0nt:{}".format(s0['nt']))
        feats.append("S0lnt:{}".format(s0['lchnt']))
        feats.append("S0rnt:{}".format(s0['rchnt']))
        feats.append("S0nch:{}".format(s0['nch']))
        feats.append("S0nlch:{}".format(s0['nlch']))
        feats.append("S0nrch:{}".format(s0['nrch']))

        # features of the 1st item on the stack
        for word in s1['head']:
            feats.append("S1w:{}".format(word))
        for pos_tag in s1['hpos']:
            feats.append("S1p:{}".format(pos_tag))
        feats.append("S1nt:{}".format(s1['nt']))
        feats.append("S1lnt:{}".format(s1.get('lchnt', '')))
        feats.append("S1rnt:{}".format(s1.get('rchnt', '')))
        feats.append("S1nch:{}".format(s1.get('nch', '')))
        feats.append("S1nlch:{}".format(s1.get('nlch', '')))
        feats.append("S1nrch:{}".format(s1.get('nrch', '')))

        # features of the 2nd item on the stack
        for word in s2['head']:
            feats.append("S2w:{}".format(word))
        for pos_tag in s2['hpos']:
            feats.append("S2p:{}".format(pos_tag))
        feats.append("S2nt:{}".format(s2['nt']))

        # features of the 3rd item on the stack
        feats.append("S3nt:{}".format(s3['nt']))

        # features for the next items on the input queue
        for word in nw1:
            feats.append("nw1:{}".format(word))
        for pos_tag in np1:
            feats.append("np1:{}".format(pos_tag))
        for word in nw2:
            feats.append("nw2:{}".format(word))
        for pos_tag in np2:
            feats.append("np2:{}".format(pos_tag))

        # distance feature
        # TODO do these thresholds for distance features need to be adjusted?
        dist = s0.get('idx', 0) - s1.get('idx', 0)
        if dist > 10:
            dist = 10
        if dist > 7 and dist != 10:
            dist = 7
        feats.append("dist:{}".format(dist))

        # combinations of features
        # TODO re-enable combo features?
        # for i in range(len(feats)):
        #     feats.append("combo:{}~PREV:{}".format(feats[i], prevact))
        #     feats.append("combo:{}~S0p:{}".format(feats[i],
        #                                           s0['hpos'][1]
        #                                           if len(s0['hpos']) > 1
        #                                           else ""))
        #     feats.append("combo:{}~np1:{}".format(feats[i],
        #                                           np1[1]
        #                                           if len(np1) > 1
        #                                           else ""))
        return feats

    @staticmethod
    def is_valid_action(act, ucnt, sent, stack):
        if act.startswith("U"):
            # Don't allow too many consecutive unary reduce actions.
            if ucnt > 2:
                return False

            # Don't allow a reduce action if the stack is empty.
            # (i.e., contains only the leftwall)
            if stack[-1]["head"] == "LEFTWALL":
                return False

            # Don't allow unary reduces on internal nodes for binarized rules.
            if stack[-1]["nt"].endswith('*'):
                return False

        # Don't allow shift if there is nothing left to shift.
        if act.startswith("S") and not sent:
            return False

        # Don't allow a binary reduce unless there are at least two items in
        # the stack to be reduced (plus the leftwall),
        # with one of them being a nucleus or a partial subtree containing
        # a nucleus, as indicated by a * suffix).
        if re.search(r'^[RL]', act) and act != "R:ROOT":
            # Make sure there are enough items to reduce
            # (including the left wall).
            if len(stack) < 3:
                return False

            # Make sure there is a head.
            lc_label = stack[-2]['nt']
            rc_label = stack[-1]['nt']
            if not (lc_label.startswith('nucleus')
                    or rc_label.startswith('nucleus')
                    or lc_label.endswith('*')
                    or rc_label.endswith('*')):
                return False

            # Check that partial node labels (ending with *) match the action.
            act_label = act[2:]
            if lc_label.endswith('*') \
                    and act_label != lc_label and act_label != lc_label[:-1]:
                return False
            if rc_label.endswith('*') \
                    and act_label != rc_label and act_label != rc_label[:-1]:
                return False

        # Don't allow R:ROOT unless we will have a complete parse.
        if act == "R:ROOT" and (len(stack) != 2 or sent):
            return False

        # Default: the action is valid.
        return True

    @staticmethod
    def process_action(act, sent, stack):
        # The R action reduces the stack, creating a non-terminal node
        # with a lexical head coming from the left child
        # (this is a confusing name, but it refers to the direction of
        # the dependency arrow).
        match = re.search(r'^R:(.+)$', act)
        if match:
            label = match.groups()[0]

            tmp_rc = stack.pop()
            tmp_lc = stack.pop()
            new_tree = ParentedTree("({})".format(label))
            new_tree.append(tmp_lc['tree'])
            new_tree.append(tmp_rc['tree'])

            tmp_item = {"idx": tmp_lc["idx"],
                        "nt": label,
                        "tree": new_tree,
                        "head": tmp_lc["head"],
                        "hpos": tmp_lc["hpos"],
                        "lchnt": tmp_lc["lchnt"],
                        "rchnt": tmp_rc["nt"],
                        "lchpos": tmp_lc["lchpos"],
                        "rchpos": tmp_rc.get("pos", ""),
                        "lchw": tmp_lc["lchw"],
                        "rchw": tmp_rc["head"],
                        "nch": tmp_lc["nch"] + 1,
                        "nlch": tmp_lc["nlch"] + 1,
                        "nrch": tmp_lc["nrch"]}
            stack.append(tmp_item)

        # The L action is like the R action but with lexical head
        # coming from left child.
        match = re.search(r'^L:(.+)$', act)
        if match:
            label = match.groups()[0]

            tmp_rc = stack.pop()
            tmp_lc = stack.pop()

            new_tree = ParentedTree("({})".format(label))
            new_tree.append(tmp_lc["tree"])
            new_tree.append(tmp_rc["tree"])

            tmp_item = {"idx": tmp_rc["idx"],
                        "nt": label,
                        "tree": new_tree,
                        "head": tmp_rc["head"],
                        "hpos": tmp_rc["hpos"],
                        "lchnt": tmp_lc["nt"],
                        "rchnt": tmp_rc["rchnt"],
                        "lchpos": tmp_lc.get("pos", ""),
                        "rchpos": tmp_rc["rchpos"],
                        "lchw": tmp_lc["head"],
                        "rchw": tmp_rc["rchw"],
                        "nch": tmp_rc["nch"] + 1,
                        "nlch": tmp_rc["nlch"],
                        "nrch": tmp_rc["nrch"] + 1}
            stack.append(tmp_item)

        # The U action creates a unary chain (e.g., "(NP (NP ...))").
        match = re.search(r'^U:(.+)$', act)
        if match:
            nt = match.groups()[0]

            tmp_c = stack.pop()
            new_tree = ParentedTree("({})".format(nt))
            new_tree.append(tmp_c["tree"])
            tmp_item = {"idx": tmp_c["idx"],
                        "nt": nt,
                        "tree": new_tree,
                        "head": tmp_c["head"],
                        "hpos": tmp_c["hpos"],
                        "lchnt": tmp_c["lchnt"],
                        "rchnt": tmp_c["rchnt"],
                        "lchpos": tmp_c["lchpos"],
                        "rchpos": tmp_c["rchpos"],
                        "lchw": tmp_c["lchw"],
                        "rchw": tmp_c["rchw"],
                        "nch": tmp_c["nch"],
                        "nlch": tmp_c["nlch"],
                        "nrch": tmp_c["nrch"]}
            stack.append(tmp_item)

        # The S action gets the next input token
        # and puts it on the stack.
        match = re.search(r'^S:(.+)$', act)
        if match:
            stack.append(sent.pop(0))

    @staticmethod
    def initialize_edu_data(edus):
        '''
        Create a representation of the list of EDUS that make up the input.
        '''

        wnum = 0  # counter for distance features
        res = []
        for edu_index, edu in enumerate(edus):
            edu_words = [x[0] for x in edu]
            edu_pos_tags = [x[1] for x in edu]
            edustr = ' '.join(edu_words)

            # TODO move the chunk of code immediately below to mkfeats?
            # This adds special tokens for the first two words and last
            # word. These are used when computing features later. It would
            # probably be better to do this in the feature extraction code
            # rather than here.
            # The ":::N" part is just a special marker to distinguish these
            # from regular word tokens.
            edu_words.insert(0, '{}:::1'.format(edu_words[1]
                                                if len(edu_words) > 1
                                                else ""))
            edu_words.insert(0, '{}:::0'.format(edu_words[1]))
            edu_words.insert(0, '{}:::-1'.format(edu_words[-1]))
            edu_pos_tags.insert(0, '{}:::1'.format(edu_pos_tags[1]
                                                   if len(edu_pos_tags) > 1
                                                   else ""))
            edu_pos_tags.insert(0, '{}:::0'.format(edu_pos_tags[1]))
            edu_pos_tags.insert(0, '{}:::-1'.format(edu_pos_tags[-1]))

            # make a dictionary for each EDU
            wnum += 1
            new_tree = ParentedTree('(text)')
            new_tree.append(edu_index)
            tmp_item = {'idx': wnum,
                        'nt': "EDU",
                        'head': edu_words,
                        'hpos': edu_pos_tags,
                        'tree': new_tree,
                        'lchnt': "NONE",
                        'rchnt': "NONE",
                        'lchpos': "NONE",
                        'rchpos': "NONE",
                        'lchw': "NONE",
                        'rchw': "NONE",
                        'nch': 0,
                        'nlch': 0,
                        'nrch': 0}
            res.append(tmp_item)
        return res

    @staticmethod
    def deep_copy_stack_or_queue(data_list):
        res = [dict((key, val.copy(deep=True)) if key == 'tree'
                    else (key, deepcopy(val))
                    for key, val in list_item.items())
               for list_item in data_list]
        return res

    def parse(self, edus, gold_actions=None):
        '''
        `edus` is a list of (word, pos) tuples.
        
        If `gold_actions` is specified, then the parser will behave as if in 
        training mode.
        '''
        logging.info('RST parsing document...')

        states = []
        completetrees = []

        sent = self.initialize_edu_data(edus)

        # initialize the stack
        stack = []

        # TODO make stack items namedtuples
        tmp_item = {'idx': 0,
                    'nt': "LEFTWALL",
                    'tree': ParentedTree('(LEFTWALL)'),
                    'head': ["LEFTWALL"],
                    'hpos': ["LW"],
                    'lchnt': "NONE",
                    'rchnt': "NONE",
                    'lchpos': "NONE",
                    'rchpos': "NONE",
                    'lchw': "NONE",
                    'rchw': "NONE",
                    'nch': 0,
                    'nlch': 0,
                    'nrch': 0}
        stack.append(tmp_item)

        prevact = "S"
        ucnt = 0  # number of consecutive unary reduce actions

        # insert an initial state on the state list
        # TODO make state items namedtuples
        tmp_state = {"prevact": prevact,
                     "ucnt": 0,
                     "score": 0.0,  # log probability
                     "nsteps": 0,
                     "stack": stack,
                     "sent": sent}
        states.append(tmp_state)

        # loop while there are states to process
        while states:
            states.sort(key=itemgetter('score'), reverse=True)
            states = states[:self.max_states]

            cur_state = states.pop(0)  # should maybe replace this with a deque
            logging.debug("cur_state score: {}, num. states: {}".format(cur_state["score"], len(states)))

            # check if the current state corresponds to a complete tree
            if len(cur_state["sent"]) == 0 and len(cur_state["stack"]) == 1:
                tree = cur_state["stack"][0]["tree"]

                # remove the dummy LEFTWALL node
                assert tree[0].label() == 'LEFTWALL'
                del tree[0]

                # collapse binary branching * rules in the output
                collapse_binarized_nodes(tree)

                completetrees.append({'tree': tree,
                                      'score': cur_state["score"]})
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
            feats = self.mkfeats(prevact, sent, stack)

            # Compute the possible actions given this state.
            # During training, print them out.
            # During parsing, score them according to the model and sort.
            scored_acts = []
            if gold_actions is not None:
                # take the next action from gold_actions
                action_str = gold_actions.pop(0) if gold_actions else ''
                if not action_str:
                    logger.error('Ran out of gold actions for state %s and ' +
                                 'gold_actions %s', cur_state, gold_actions)
                    break

                action_str = re.sub(r'^S:(.*)$', r'S:POS', action_str)

                if not (action_str == cur_state["prevact"] and
                        action_str.startswith('U')):
                    yield (action_str, feats)

                scored_acts.append(ScoredAction(action_str, 1))
            else:
                vectorizer = self.model.feat_vectorizer
                examples = skll.data.ExamplesTuple(None, None,
                                                   vectorizer.transform(Counter(feats)),
                                                   vectorizer)
                scores = [np.log(x) for x in self.model.predict(examples)[0]]
                scored_acts = sorted(zip(self.model.label_list, scores),
                                     key=itemgetter(1),
                                     reverse=True)
                #print('\n'.join(['{} {:.4g}'.format(x.action, x.score) for x in scored_acts]), file=sys.stderr)
                #print('\n', file=sys.stderr)

            # If parsing, verify the validity of the actions.
            if gold_actions is None:
                scored_acts = [x for x in scored_acts
                               if self.is_valid_action(x[0], ucnt, sent, stack)]

            # Don't exceed the maximum number of actions
            # to consider for a parser state.
            scored_acts = scored_acts[:self.max_acts]

            while scored_acts:
                # Make deep copies of the input queue and stack.
                sent = self.deep_copy_stack_or_queue(cur_state["sent"])
                stack = self.deep_copy_stack_or_queue(cur_state["stack"])
                prevact = cur_state["prevact"]
                ucnt = cur_state["ucnt"]

                action, score = scored_acts.pop(0)

                # If the action is a unary reduce, increment the count.
                # Otherwise, reset it.
                ucnt = ucnt + 1 if action.startswith("U") else 0

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
            new_tree = ParentedTree("(ROOT)")
            for i in range(len(edus)):
                tmp_child = ParentedTree('(text)')
                tmp_child.append(i)
                new_tree.append(tmp_child)
            completetrees.append({'tree': new_tree, 'score': 0.0})

        if gold_actions is None:
            for t in completetrees:
                yield t
