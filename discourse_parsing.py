#!/usr/bin/env python3

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

# TO PARSE:
# perl slpar.pl -m MODELFILE INPUTFILE
#
# TO TRAIN:
# Not available yet.

'''

import re
from collections import defaultdict
from operator import itemgetter

import numpy as np


class Parser(object):
    def __init__(self, max_acts, max_states, n_best):
        self.max_acts = max_acts
        self.max_states = max_states
        self.n_best = n_best
        self.weights = None

    def set_weights(self, weights):
        self.weights = weights

    def mxclassify(self, feats):
        '''
        do maximum entropy classification using weight vector
        '''
        z = 0.0
        scores = defaultdict(float)
        for category in self.weights.keys():
            for feat in feats:
                if feat in self.weights[category]:
                    scores[category] += self.weights[category][feat]

            z += np.exp(scores[category])

        # divide all by z to make a distribution
        for category in self.weights.keys():
            scores[category] = np.exp(scores[category]) / z

        return scores

    @staticmethod
    def mkfeats(prevact, sent, stack):
        '''
        get features of the parser state represented
        by the current stack and queue
        '''

        nw1 = "RightWall"
        nw2 = "RightWall"
        # nw3 = "RightWall"

        np1 = "RW"
        np2 = "RW"
        # np3 = "RW"

        s0 = stack[-1]
        s1 = {'nt': "TOP", 'head': "LeftWall", 'hpos': "LW", 'tree': []}
        s2 = {'nt': "TOP", 'head': "LeftWall", 'hpos': "LW", 'tree': []}
        s3 = {'nt': "TOP", 'head': "LeftWall", 'hpos': "LW", 'tree': []}

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
        for pos_tag in s3['hpos']:
            feats.append("S3p:{}".format(pos_tag))
        feats.append("S3nt:{}".format(s3['nt']))

        # TODO 
        for word in nw1:
            feats.append("nw1:{}".format(word))
        for pos_tag in np1:
            feats.append("np1:{}".format(pos_tag))
        for word in nw2:
            feats.append("nw2:{}".format(word))
        for pos_tag in np2:
            feats.append("np2:{}".format(pos_tag))

        # distance feature
        # TODO do these thresholds need to be adjusted?
        dist = s0.get('idx', 0) - s1.get('idx', 0)  # TODO is it OK to assume 0 if key is not in dictionary?
        if dist > 10:
            dist = 10
        if dist > 7 and dist != 10:
            dist = 7
        feats.append("dist:{}".format(dist))

        # combinations of features
        nf = len(feats)
        for i in range(nf):
            feats.append("combo:{}~PREV:{}".format(feats[i], prevact))
            feats.append("combo:{}~np1:{}".format(feats[i], np1[0]))  # TODO is this the right index for np1?
            feats.append("combo:{}~S0p:{}".format(feats[i], s0['hpos'][0])) # TODO is this the right index for a11?

        return feats

    @staticmethod
    def is_valid_action(act, ucnt, sent, stack):
        # don't allow too many consecutive unary reduce actions
        if act.startswith("U") and ucnt > 2:
            return False

        # don't allow a reduce action if the stack is empty
        # (contains only the leftwall)
        if act.startswith("U") and stack[-1]["head"] == "LEFTWALL":
            return False

        # don't allow shift if there is nothing left to shift
        if act.startswith("S") and not sent:
            return False

        # don't allow a reduce right or left if there are not
        # at least two items in the stack to be reduced
        # (plus the leftwall)
        if re.search(r'^[RL]', act) \
                and act != "R:ROOT" and len(stack) < 3:
            return False
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
            new_tree = "({} {} {})".format(label,
                                           tmp_lc["tree"],
                                           tmp_rc["tree"])
            if label.endswith("*") or label == "ROOT":
                new_tree = "{} {}".format(tmp_lc["tree"],
                                          tmp_rc["tree"])

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

            new_tree = "({} {} {})".format(label,
                                           tmp_lc["tree"],
                                           tmp_rc["tree"])
            if label.endswith("*") or label == "ROOT":
                new_tree = "{} {}".format(tmp_lc["tree"],
                                          tmp_rc["tree"])

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
                        "nlch": tmp_lc["nlch"],
                        "nrch": tmp_lc["nrch"] + 1}
            stack.append(tmp_item)

        # The U action creates a unary chain (e.g., "(NP (NP ...))").
        match = re.search(r'^U:(.+)$', act)
        if match:
            nt = match.groups()[0]

            tmp_c = stack.pop()
            tmp_item = {"idx": tmp_c["idx"],
                        "nt": nt,
                        "tree": "({} {})".format(nt, tmp_c["tree"]),
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
            #pos = match.groups()[0]  # TODO was this meant for something or left over from the constituency parser?
            stack.append(sent.pop(0))

    def parse(self, edus, train_mode=False):
        '''
        edus is a list of (word, pos) tuples
        '''

        gold_acts = []
        sts = []
        completetrees = []
        sent = []

        wnum = 0  # TODO should this be a member variable?

        # create a representation of the list of EDUS that make up the input
        for edu in edus:
            edu_words = [x[0] for x in edu]
            edu_pos_tags = [x[1] for x in edu]
            edustr = ' '.join(edu_words)

            # TODO move this to mkfeats?
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
            tmp_item = {'idx' : wnum,
                        'nt' : edu_pos_tags[-1],  # TODO why was this $2 in the perl code?
                        'head' : edu_words,
                        'hpos' : edu_pos_tags,
                        'tree' : "(text _!{}!_)".format(edustr),
                        '#tree' : "(EDU {})".format(wnum),
                        'lchnt' : "NONE",
                        'rchnt' : "NONE",
                        'lchpos' : "NONE",
                        'rchpos' : "NONE",
                        'lchw' : "NONE",
                        'rchw' : "NONE",
                        'nch' : 0,
                        'nlch' : 0,
                        'nrch' : 0}
            sent.append(tmp_item)

        # if we are training, the gold actions should be
        # in the input file
        if train_mode:
            # TODO
            #     <>;    #blank line
            #     my $actstr = <>;
            #     $actstr =~ s/[\n\r]//g;
            #     # put the gold actions in @goldacts
            #     @goldacts = split /[ \t]+/, $actstr;
            #     <>;    #blank line
            pass

        # initialize the stack
        stack = []
        tmp_item = {'idx' : 0,
                    'nt' : "LEFTWALL",
                    'tree' : "",
                    'head' : ["LEFTWALL"],
                    'hpos' : ["LW"],
                    'lchnt' : "NONE",
                    'rchnt' : "NONE",
                    'lchpos': "NONE",
                    'rchpos': "NONE",
                    'lchw' : "NONE",
                    'rchw' : "NONE",
                    'nch' : 0,
                    'nlch' : 0,
                    'nrch' : 0}
        stack.append(tmp_item)

        prevact = "S"
        ucnt = 0
        initialsent = sent

        # insert an initial state on the state list
        tmp_state = {"prevact": prevact,
                     "ucnt": 0,
                     "score": 1,
                     "nsteps": 0,
                     "stack": stack,
                     "sent": sent}
        sts.append(tmp_state)

        # loop while there are states to process
        while sts:
            sts.sort(key=lambda x: x['score'], reverse=True)
            if len(sts) > self.max_states:
                sts = sts[:self.max_states]

            cur_state = sts.pop(0)  # should maybe replace this with a deque

            if len(cur_state["sent"]) == 0 and len(cur_state["stack"]) == 1:
                # check if the current state corresponds to a complete tree
                completetrees.append({'tree': cur_state["stack"][0]["tree"],
                                      'score': cur_state["score"]})
                if train_mode or len(completetrees) >= self.n_best:
                    break

            stack = cur_state["stack"]
            sent = cur_state["sent"]
            prevact = cur_state["prevact"]
            ucnt = cur_state["ucnt"]

            # extract features
            feats = self.mkfeats(prevact, sent, stack)

            acts = []

            # Compute the possible actions given this state.
            # During training, print them out.
            # During parsing, score them according to the model and sort.
            if train_mode:
                pass
                # TODO
                # # take the next action from @goldacts
                # my $tmpstr = shift @goldacts;
                # $acts[0] = {
                #     act   => $tmpstr,
                #     score => 1,
                # };
                # if ( $acts[0]->{act} eq "" ) {
                #     $numparseerror++;
                #     print STDERR
                #         "Parse error (no more actions). $numparseerror\n";
                #     last;
                # }

                # $acts[0]->{act} =~ s/^S:(.*)$/S:POS/;
                # $acts[0]->{act} =~ s/^([^\=\-]+)[\=\-].+/$1/;

                # if (!(     ( $acts[0]->{act} eq $prevact )
                #         && ( $acts[0]->{act} =~ /^U/ )
                #     )
                #     )
                # {
                #     my $featstr = join " ", @{$featsref};
                #     print "$acts[0]->{act} $featstr\n";
                # }
            else:
                acts = self.mxclassify(feats)
                acts = sorted(acts.items(), key=itemgetter(1), reverse=True)

            nacts = 0
            while acts:
                stack = cur_state["stack"]
                sent = cur_state["sent"]
                prevact = cur_state["prevact"]
                ucnt = cur_state["ucnt"]

                act, score = acts.pop(0)

                # If parsing, verify the validity of the action.
                if not train_mode:
                    if not self.is_valid_action(act, ucnt, sent, stack):
                        continue

                if not train_mode:
                    # If the action is a unary reduce, increment the count.
                    # Otherwise, reset it.
                    ucnt = ucnt + 1 if act.startswith("U") else 0

                # Don't exceed the maximum number of actions
                # to consider for a parser state.
                nacts += 1
                if nacts > self.max_acts:
                    break

                self.process_action(act, sent, stack)

                # Add the newly created state
                tmp_state = {"prevact": act,
                             "ucnt": ucnt,
                             "score": cur_state["score"] * score,
                             "nsteps": cur_state["nsteps"] + 1,
                             "stack": stack,
                             "sent": sent}
                sts.append(tmp_state)

        # Done parsing.  Print the result(s).
        if not train_mode:
            if self.n_best > 1:
                for tree in completetrees:
                    print(tree["score"])
                    print("(TOP {})".format(tree["tree"]))
                print()
            else:
                if completetrees:
                    print("(TOP {})".format(completetrees[0]["tree"]))
                else:
                    # Default to a flat tree if there is no complete parse.
                    tmp_str = ""
                    for e in initialsent:
                        tmp_str += " (text {})".format(" ".join(e["head"]))
                    print("(TOP {})".format(tmp_str))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-m', '--modelpath')
    parser.add_argument('input_path', help='file to parse, with one EDU \
                         per line, with POS tags \
                         (e.g., "This/DT is/VBZ a/DT test/NN ./.").')
    parser.add_argument('--max_states', type=int, default=50)
    parser.add_argument('--max_acts', type=int, default=5)
    parser.add_argument('--n_best', type=int, default=1)
    args = parser.parse_args()

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=args.n_best)

    if not args.train:
        # read the model
        weights = defaultdict(dict)
        with open(args.modelpath) as model_file:
            for line in model_file:
                parts = line.strip().split()
                weights[parts[0]][parts[1]] = float(parts[2])
        parser.set_weights(weights)

    with open(args.input_path) as f:
        data = f.read().strip()
        docs = re.split(r'\n\n+', data)

        for doc in docs:
            # Split the document into edus, one edu per line (with POS tags)
            # e.g., This/DT is/VBZ a/DT test/NN ./."
            # TODO change this to read in the JSON format that also includes
            # PTB trees.
            edus = []
            for edu_str in doc.split("\n"):
                edu = []
                for tagged_token in edu_str.strip().split():
                    slash_idx = tagged_token.rindex('/')
                    edu.append((tagged_token[:slash_idx],
                                tagged_token[slash_idx + 1:]))
                edus.append(edu)

            tree = parser.parse(edus)


if __name__ == '__main__':
    main()
