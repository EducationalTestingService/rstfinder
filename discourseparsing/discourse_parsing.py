"""
A Python shift-reduce RST discourse parser.

This code is based partly on a Perl parser written by Kenji Sagae.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import itertools
import logging
import re
from collections import Counter, namedtuple
from operator import itemgetter
from os.path import join

import numpy as np
from nltk.tree import ParentedTree, Tree
from skll.data import FeatureSet
from skll.learner import Learner

from .discourse_segmentation import extract_tagged_doc_edus
from .tree_util import HeadedParentedTree, collapse_binarized_nodes

ShiftReduceAction = namedtuple("ShiftReduceAction", ["type", "label"])
ScoredAction = namedtuple("ScoredAction", ["action", "score"])
logger = logging.getLogger(__name__)


class Parser(object):
    """
    Class representing the RST discourse parser.

    This parser follows an arc-standard parsing strategy, where reduce actions
    operate on the top 2 stack items cf. Nivre (2004).

    Joakim Nivre. 2004. Incrementality in Deterministic Dependency Parsing.
    Proc. of the Workshop on Incremental Parsing.
    """

    leftwall_w = "LEFTWALL"
    leftwall_p = "LEFTWALL"
    rightwall_w = "RIGHTWALL"
    rightwall_p = "RIGHTWALL"
    max_consecutive_unary_reduce = 2

    def __init__(self, max_acts, max_states, n_best):
        """
        Initialize the parser.

        Parameters
        ----------
        max_acts : int
            Maximum number of actions to perform on each state.
        max_states : int
            Maximum number of states to retain for best-first search.
        n_best : int
            Number of RST parses to return.
        """
        self.max_acts = max_acts
        self.max_states = max_states
        self.n_best = n_best
        self.model = None
        self.model_action_list = None

    def load_model(self, model_path):
        """
        Load the parser model from disk.

        Parameters
        ----------
        model_path : str
            Path to the directory containing the model file.
            The model name is always
            ``rst_parsing_all_feats_LogisticRegression.model``.
        """
        model_name = "rst_parsing_all_feats_LogisticRegression.model"
        self.model = Learner.from_file(join(model_path, model_name))

    def _get_model_actions(self):
        """
        Create ``ShiftReduceAction`` objects for classifier labels.

        This is used later when parsing, to decide which
        action to take based on a list of scores.
        """
        if self.model_action_list is None:
            self.model_action_list = []
            for label_str in self.model.label_list:
                act = ShiftReduceAction(type=label_str[0], label=label_str[2:])
                self.model_action_list.append(act)
        return self.model_action_list

    @staticmethod
    def _add_word_and_pos_feats(feats, prefix, words, pos_tags):
        """
        Add word and POS features to the head EDU of a subtree.

        This function also adds specially marked features for the first 2
        and last 1 token. ``feats`` is the existing list of features.
        The prefix indicates where the tokens are from (S0, S1, S2, Q0, Q1).

        Parameters
        ----------
        feats : list
            List of existing features.
        prefix : str
            Indicates where the tokens are from (S0, S1, S2, Q0, Q1).
        words : list
            List of tokens.
        pos_tags : list
            List of part-of-speech tags.
        """
        # Do not add any word or POS features for the LEFTWALL or RIGHTWALL.
        # That information should be available in the nonterminal features.
        if pos_tags == [Parser.leftwall_p] or pos_tags == [Parser.rightwall_p]:
            assert (words == [Parser.leftwall_w] or
                    words == [Parser.rightwall_w])
        else:
            # first 2 and last 1
            feats.append(f"{prefix}w:{words[0]}:::0")
            feats.append(f"{prefix}p:{pos_tags[0]}:::0")
            feats.append(f"{prefix}w:{words[-1]}:::-1")
            feats.append(f"{prefix}p:{pos_tags[-1]}:::-1")
            word_feature = words[1] if len(words) > 1 else ""
            feats.append(f"{prefix}w:{word_feature}:::1")
            pos_tag_feature = pos_tags[1] if len(pos_tags) > 1 else ""
            feats.append(f"{prefix}p:{pos_tag_feature}:::1")

            for word in words:
                feats.append(f"{prefix}w:{word}")
            for pos_tag in pos_tags:
                feats.append(f"{prefix}p:{pos_tag}")

    @staticmethod
    def _find_edu_head_node(rst_node, doc_dict):
        """
        Find the EDU head node.

        The EDU head node is the node whose head is the word with the
        highest occurrence as a lexical head (Soricut & Marco, 2003,
        Sec 4.1).

        Note that there can be ties, which the paper doesn't mention.
        This function just finds the leftmost, using np.argmin on tree depths.

        Parameters
        ----------
        rst_node : dict
            The dictionary representing the RST node.
        doc_dict : dict
            The dictionary representing the document.

        Returns
        -------
        result : nltk.tree.ParentedTree
            The object representing the EDU head node.
        """
        # return None for the left wall
        head_idx = rst_node["head_idx"]
        if head_idx is None:
            return None

        head_words = rst_node["head"]

        edu_start_indices = doc_dict['edu_start_indices'][head_idx]
        tree_idx, start_tok_idx, _ = edu_start_indices
        tree = doc_dict['syntax_trees_objs'][tree_idx]
        end_tok_idx = start_tok_idx + len(head_words)
        preterminals = [x for x in tree.subtrees()
                        if isinstance(x[0], str)][start_tok_idx:end_tok_idx]

        # filter out punctuation if the EDU has more than just punctuation.
        # otherwise "." will be the head of sentences.
        filtered_preterminals = [node for node in preterminals
                                 if re.search(r'[A-Za-z]', node.label())]
        if not filtered_preterminals:
            logging.debug((f"EDU head only contained punctuation: {preterminals}, "
                           f"doc_id = {doc_dict['doc_id']}"))
            return None

        preterminals = filtered_preterminals
        maximal_nodes = [node.find_maximal_head_node() for node in preterminals]
        depths = [len(node.treeposition()) for node in maximal_nodes]
        mindepth_idx = np.argmin(depths)
        res = maximal_nodes[mindepth_idx]
        return res

    @staticmethod
    def syntactically_dominates(node1, node2):
        """
        Check if ``node1`` is an ancestor of ``node2``.

        Returns ``True`` if the two nodes are in the same tree and ``node1``
        is an ancestor of ``node2``.

        Parameters
        ----------
        node1 : nltk.tree.ParentedTree
            The first node.
        node2 : nltk.tree.ParentedTree
            The second node.

        Returns
        -------
        dominates : bool
            ``True`` if ``node1`` is an ancestor of ``node2``,
            ``False`` otherwise.
        """
        if (node1 is None or
                node2 is None or
                node1.root() != node2.root()):
            return False
        tree_position_node1 = node1.treeposition()
        tree_position_node1 = node2.treeposition()

        # return `False` if node 1 is deeper in the tree
        if len(tree_position_node1) >= len(tree_position_node1):
            return False

        # the treeposition (i.e., sequence of child indices from the root)
        # of `node1` should be a prefix of the treeposition of `node2`
        res = (tree_position_node1 == tree_position_node1[:len(tree_position_node1)])
        return res

    @staticmethod
    def mkfeats(state, doc_dict):
        """
        Get features from current parser state.

        The current parser state is represented by the current stack and queue.

        Parameters
        ----------
        state : dict
            The dictionary representing the current state.
        doc_dict : dict
            The dictionary representing the document.

        Returns
        -------
        feats : list
            The list of features for the current parser state.
        """
        feats = []

        # initialize some local variables for top stack and next queue items
        prevact = state["prevact"]
        queue = state["queue"]
        stack = state["stack"]

        s0 = {"nt": "TOP",
              "head": [Parser.leftwall_w],
              "hpos": [Parser.leftwall_p],
              "tree": None,
              "head_idx": None,
              "start_idx": None}
        s1 = {"nt": "TOP",
              "head": [Parser.leftwall_w],
              "hpos": [Parser.leftwall_p],
              "tree": None,
              "head_idx": None,
              "start_idx": None}
        s2 = {"nt": "TOP",
              "head": [Parser.leftwall_w],
              "hpos": [Parser.leftwall_p],
              "tree": None,
              "head_idx": None,
              "start_idx": None}

        stack_len = len(stack)
        if stack_len > 0:
            s0 = stack[-1]
        if stack_len > 1:
            s1 = stack[-2]
        if stack_len > 2:
            s2 = stack[-3]

        q0w = [Parser.rightwall_w]
        q0p = [Parser.rightwall_p]
        if len(queue) > 0:
            q0w = queue[0]["head"]
            q0p = queue[0]["hpos"]

        # make a list of head EDU idx tuples for the top stack/queue items;
        # filter out `None` for when the stack/queue does not have that many
        # items
        s0_idx = s0["head_idx"]
        s1_idx = s1["head_idx"]
        s2_idx = s2["head_idx"]
        q0_idx = queue[0]["head_idx"] if queue else None
        label_idx_tuples = [('Q0', q0_idx), ('S0', s0_idx),
                            ('S1', s1_idx), ('S2', s2_idx)]
        label_idx_tuples = [x for x in label_idx_tuples if x[1] is not None]

        # previous action feature
        feats.append(f"PREV:{prevact.type}:{prevact.label}")

        # stack nonterminal symbol features
        feats.append(f"S0nt:{s0['nt']}")
        if s0["tree"] is not None and s0["tree"].label() != "text":
            for child in s0["tree"]:
                feats.append(f"S0childnt:{child.label()}")

        feats.append(f"S1nt:{s1['nt']}")
        if s1["tree"] is not None and s1["tree"].label() != "text":
            for child in s1["tree"]:
                feats.append(f"S1childnt:{child.label()}")

        feats.append(f"S2nt:{s2['nt']}")
        if s2["tree"] is not None and s2["tree"].label() != "text":
            for child in s2["tree"]:
                feats.append(f"S2childnt:{child.label()}")

        feats.append(f"S0nt:{s0['nt']}^S1nt:{s1['nt']}")
        feats.append(f"S1nt:{s1['nt']}^S2nt:{s2['nt']}")
        feats.append(f"S0nt:{s0['nt']}^S2nt:{s2['nt']}")
        feats.append(f"S0nt:{s0['nt']}^S1nt:{s1['nt']}^S2nt:{s2['nt']}")

        # features for the words and POS tags of the heads of the first and
        # last tokens of the heads of the top stack and next input queue items
        Parser._add_word_and_pos_feats(feats, 'S0', s0["head"], s0["hpos"])
        Parser._add_word_and_pos_feats(feats, 'S1', s1["head"], s1["hpos"])
        Parser._add_word_and_pos_feats(feats, 'Q0', q0w, q0p)

        # EDU head distance feature
        # (this is in EDUs, not sentences or tokens. None is for the left wall)
        for ((label_a, idx_a),
             (label_b, idx_b)) in itertools.combinations(label_idx_tuples, 2):
            dist = abs(idx_a - idx_b)
            for i in range(1, 5):
                if dist > i:
                    feats.append(f"edu_dist_{label_a}{label_b}>{i}")

        # whether the EDUS are in the same sentence
        # (edu_start_indices is a list of (sentence #, token #, EDU #) tuples.
        # Also, EDUs don't cross sentence boundaries.)
        start_indices = doc_dict["edu_start_indices"]
        for ((label_a, idx_a),
             (label_b, idx_b)) in itertools.combinations(label_idx_tuples, 2):
            if start_indices[idx_a][0] == start_indices[idx_b][0]:
                feats.append(f"same_sentence_{label_a}{label_b}")

        # features of EDU heads
        head_node_s0 = Parser._find_edu_head_node(s0, doc_dict)
        head_node_s1 = Parser._find_edu_head_node(s1, doc_dict)
        head_node_s2 = Parser._find_edu_head_node(s2, doc_dict)
        head_node_q0 = Parser._find_edu_head_node(queue[0], doc_dict) if queue else None
        if head_node_s0 is not None:
            feats.append(f"S0headnt:{head_node_s0.label()}")
            feats.append(f"S0headw:{head_node_s0.head_word().lower()}")
            feats.append(f"S0headp:{head_node_s0.head_pos()}")
        if head_node_s1 is not None:
            feats.append(f"S1headnt:{head_node_s1.label()}")
            feats.append(f"S1headw:{head_node_s1.head_word().lower()}")
            feats.append(f"S1headp:{head_node_s1.head_pos()}")
        if head_node_q0 is not None:
            feats.append(f"Q0headnt:{head_node_q0.label()}")
            feats.append(f"Q0headw:{head_node_q0.head_word().lower()}")
            feats.append(f"Q0headp:{head_node_q0.head_pos()}")

        # syntactic dominance features between pairs of stack/queue items:
        # (This is similar to Feng & Hirst, ACL 2014, and also vaguely similar
        # to Soricut & Marcu, 2003.)
        label_node_tuples = [("Q0", head_node_q0), ("S0", head_node_s0),
                             ("S1", head_node_s1), ("S2", head_node_s2)]
        for ((nlabel1, node1),
             (nlabel2, node2)) in itertools.combinations(label_node_tuples, 2):
            if Parser.syntactically_dominates(node1, node2):
                feats.append(f"syn_dominates_{nlabel1}{nlabel2}")
            if Parser.syntactically_dominates(node2, node1):
                feats.append(f"syn_dominates_{nlabel2}{nlabel1}")

        # paragraph and document position features
        starts_paragraph = doc_dict["edu_starts_paragraph"]
        s0_start_idx = s0["start_idx"]
        s1_start_idx = s1["start_idx"]
        s2_start_idx = s2["start_idx"]
        q0_start_idx = queue[0]['start_idx'] if queue else None
        if s0_start_idx is not None and starts_paragraph[s0_start_idx]:
            feats.append("s0_starts_paragraph")
        if s1_start_idx is not None and starts_paragraph[s1_start_idx]:
            feats.append("s1_starts_paragraph")
        if s2_start_idx is not None and starts_paragraph[s2_start_idx]:
            feats.append("s2_starts_paragraph")
        if q0_start_idx is not None and starts_paragraph[q0_start_idx]:
            feats.append("q0_starts_paragraph")

        return feats

    @staticmethod
    def is_valid_action(act, state):
        """
        Check if the given action is valid given the current parser state.

        Parameters
        ----------
        act : ShiftReduceAction
            The object representing the given action.
        state : dict
            The dictionary representing the current parser state.

        Returns
        -------
        is_valid : bool
            ``True`` if the action is valid, ``False`` otherwise.
        """
        queue = state["queue"]
        stack = state["stack"]
        ucnt = state["ucnt"]

        if act.type == "U":
            # Do not allow too many consecutive unary reduce actions.
            if ucnt > Parser.max_consecutive_unary_reduce:
                return False

            # Do not allow a reduce action if the stack is empty.
            if not stack:
                return False

            # Do not allow unary reduces on internal nodes for binarized rules.
            if stack[-1]["nt"].endswith('*'):
                return False

            # Do not allow unary reduce actions on satellites.
            if stack[-1]["nt"].startswith('satellite'):
                return False

            # Do not allow reduction to satellites if the queue is empty
            # and there isn't a nucleus next on the stack.
            if (act.label.startswith('satellite') and
                    not queue and
                    not stack[-2]["nt"].startswith('nucleus') and
                    not stack[-2]["nt"].endswith('*')):
                return False

        # Do not allow shift if there is nothing left to shift.
        if act.type == "S" and not queue:
            return False

        # Do not allow a binary reduce unless there are at least two items in
        # the stack to be reduced (plus the leftwall),
        # with one of them being a nucleus or a partial subtree containing
        # a nucleus, as indicated by a * suffix).
        if act.type == "B":
            # Make sure there are enough items to reduce
            if len(stack) < 2:
                return False

            # Do not allow B:ROOT unless we will have a complete parse.
            if act.label == "ROOT" and len(stack) + len(queue) > 2:
                return False
            if act.label != "ROOT" and len(stack) + len(queue) == 2:
                return False

            # Make sure there is a head.
            lc_label = stack[-2]["nt"]
            rc_label = stack[-1]["nt"]
            if not (lc_label.startswith('nucleus') or
                    rc_label.startswith('nucleus') or
                    lc_label.endswith('*') or
                    rc_label.endswith('*')):
                return False

            # Check that partial node labels (ending with *) match the action.
            if (lc_label.endswith('*') and
                    act.label != lc_label and
                    act.label != lc_label[:-1]):
                return False
            if (rc_label.endswith('*') and
                    act.label != rc_label and
                    act.label != rc_label[:-1]):
                return False

            # do not allow reduction to satellites if the queue is empty
            # and there isn't a nucleus next on the stack;
            # starred (binarized) nodes can also be nuclei,
            # but two starred nodes can't be reduced.
            label_is_satellite = act.label.startswith('satellite')
            label_is_partial_head = act.label.endswith('*')
            next_is_nucleus = (stack[-3]["nt"].startswith('nucleus')
                               if len(stack) > 2 else False)
            next_is_partial_head = (stack[-3]["nt"].endswith('*')
                                    if len(stack) > 2 else False)
            if (not queue and
                    label_is_satellite and
                    not label_is_partial_head and
                    not next_is_nucleus and
                    not next_is_partial_head):
                return False
            if (not queue and
                    next_is_partial_head and
                    label_is_partial_head):
                return False

        # default: the action is valid.
        return True

    @staticmethod
    def process_action(act, state):
        """
        Process given action in the context of the current parser state.

        Parameters
        ----------
        act : ShiftReduceAction
            The object representing the given action.
        state : dict
            The dictionary representing the current parser state.

        Raises
        ------
        ValueError
            If the binary reduce action (B) involves two non-nuclei.
        """
        # The B action reduces 2 stack items, creating a non-terminal node,
        # with the head determined by nuclearity.
        stack = state["stack"]
        queue = state["queue"]

        # If the action is a unary reduce, increment the count.
        # Otherwise, reset it.
        state["ucnt"] = state["ucnt"] + 1 if act.type == "U" else 0

        if act.type == "B":
            tmp_rc = stack.pop()
            tmp_lc = stack.pop()
            new_tree = Tree.fromstring(f"({act.label})")
            new_tree.append(tmp_lc["tree"])
            new_tree.append(tmp_rc["tree"])

            left_is_nucleus = (tmp_lc["nt"].startswith('nucleus:') or
                               tmp_lc["nt"].endswith('*') or
                               (act.type == 'B' and act.label == 'ROOT'))
            right_is_nucleus = (tmp_rc['nt'].startswith("nucleus") or
                                tmp_rc["nt"].endswith('*'))

            if left_is_nucleus:
                new_head = tmp_lc["head"]
                new_hpos = tmp_lc["hpos"]
                new_head_idx = tmp_lc["head_idx"]
            elif right_is_nucleus:
                new_head = tmp_rc["head"]
                new_hpos = tmp_rc["hpos"]
                new_head_idx = tmp_rc["head_idx"]
            else:
                raise ValueError(f"Invalid binary reduce of two non-nuclei.\n"
                                 f"act = {act.type}:{act.label}\n "
                                 f"tmp_lc = {tmp_lc}\ntmp_rc = {tmp_rc}")

            tmp_item = {"head_idx": new_head_idx,
                        "start_idx": tmp_lc["start_idx"],
                        "end_idx": tmp_rc["end_idx"],
                        "nt": act.label,
                        "tree": new_tree,
                        "head": new_head,
                        "hpos": new_hpos}

            stack.append(tmp_item)

        # The U action creates a unary chain (e.g., "(NP (NP ...))").
        if act.type == "U":
            tmp_c = stack.pop()

            if tmp_c['nt'].startswith('satellite'):
                raise ValueError(f"Invalid unary reduce of a satellite.\n"
                                 f"act = {act.type}:{act.label}\n "
                                 f"tmp_c = {tmp_c}")

            new_tree = Tree.fromstring(f"({act.label})")
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
            stack.append(queue.pop(0))

    @staticmethod
    def initialize_edu_data(edus):
        """
        Create a representation of the list of EDUS that make up the input.

        Parameters
        ----------
        edus : list
            List of EDUs for which to create representations.

        Returns
        -------
        res : list of dict
            List of dictionaries containing enhanced representations.
            for each EDU in the given list.
        """
        wnum = 0  # counter for distance features
        res = []
        for edu_index, edu in enumerate(edus):
            # lowercase all words
            edu_words = [word_tag[0].lower() for word_tag in edu]
            edu_pos_tags = [word_tag[1] for word_tag in edu]

            # make a dictionary for each EDU
            new_tree = Tree.fromstring("(text)")
            new_tree.append(f"{edu_index}")
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
        """
        Parse the given document.

        If ``make_features`` and ``gold_actions`` are specified, then the parser
        will yield (action, features) tuples instead of trees
        (e.g., to produce training examples).

        This will have no effect if `gold_actions` is not provided.
        Disabling `make features` can be useful for debugging and testing.

        Parameters
        ----------
        doc_dict : dict
            Dictionary with EDU segments, parse trees, etc.
            (see ``convert_rst_discourse_tb.py``)
        gold_actions : list, optional
            List of gold standard actions. If specified, then the parser
            will behave as if in training mode.
        make_features : bool, optional
            If ``True`` and if ``gold_actions`` are specified, then the
            parser will yield (action, features) tuples instead of trees
            (e.g., to produce training examples). This will have no effect if
            `gold_actions` is not provided. Setting this to ``False`` can be
            useful for debugging and testing.
            Defaults to ``True``.

        Yields
        ------
        output : tuple or nltk.tree.ParentedTree
            List of (action, features) tuples if in training mode (
            ``make_features`` and ``gold_actions`` are both ``True``) or
            the RST tree for the current document.
        """
        doc_id = doc_dict["doc_id"]
        logging.info(f"RST parsing, doc_id = {doc_id}")

        states = []
        completetrees = []
        tagged_edus = extract_tagged_doc_edus(doc_dict)

        queue = self.initialize_edu_data(tagged_edus)

        # If there is only one item on the queue to start, then make it a
        # finished tree so that parsing will complete immediately.
        # TODO add a unit test for this.
        if len(queue) == 1:
            logging.warning(f"There was only one EDU to parse. A very simple "
                            f"tree will be returned. doc_id = {doc_id}")
            new_tree = Tree.fromstring("(ROOT)")
            new_tree.append(queue[0]["tree"])
            queue[0]["tree"] = new_tree

        # precompute syntax tree objects so this only needs to be done once
        if ("syntax_trees_objs" not in doc_dict or
                len(doc_dict["syntax_trees_objs"]) != len(doc_dict["syntax_trees"])):
            doc_dict['syntax_trees_objs'] = []
            for tree_str in doc_dict["syntax_trees"]:
                doc_dict["syntax_trees_objs"].append(
                    HeadedParentedTree.fromstring(tree_str))

        # initialize the stack
        stack = []

        prevact = ShiftReduceAction(type="S", label="text")

        # insert an initial state on the state list
        tmp_state = {"prevact": prevact,
                     "ucnt": 0,
                     "score": 0.0,  # log probability
                     "nsteps": 0,
                     "stack": stack,
                     "queue": queue}
        states.append(tmp_state)

        # loop while there are states to process
        while states:
            states.sort(key=itemgetter("score"), reverse=True)
            states = states[:self.max_states]

            cur_state = states.pop(0)  # should maybe replace this with a deque
            prevact_type = cur_state["prevact"].type
            prevact_label = cur_state["prevact"].label
            score = cur_state["score"]
            logging.debug((f"cur_state prevact = {prevact_type}:{prevact_label}, "
                           f"score = {score}, "
                           f"num_states = {len(states)}, "
                           f"doc_id = {doc_id}"))

            # check if the current state corresponds to a complete tree
            if len(cur_state["queue"]) == 0 and len(cur_state["stack"]) == 1:
                tree = cur_state["stack"][-1]["tree"]
                assert tree.label() == 'ROOT'

                # collapse binary branching * rules in the output
                output_tree = ParentedTree.fromstring(tree.pformat())
                collapse_binarized_nodes(output_tree)

                completetrees.append({"tree": output_tree,
                                      "score": cur_state["score"]})
                logging.debug(f"complete tree found, doc_id = {doc_id}")

                # stop if we have found enough trees
                if (gold_actions is not None or
                        len(completetrees) >= self.n_best):
                    break

                # otherwise, move on to the next best state
                continue

            # extract features
            feats = self.mkfeats(cur_state, doc_dict)

            # Compute the possible actions given this state.
            # During training, print them out.
            # During parsing, score them according to the model and sort.
            scored_acts = []
            if gold_actions is not None:
                # take the next action from gold_actions
                act = gold_actions.pop(0) if gold_actions else None
                if act is None:
                    logger.error(f"Ran out of gold actions for state "
                                 f"{cur_state} and gold_actions {gold_actions}")
                    break

                assert act.type != 'S' or act.label == "text"

                if make_features:
                    if not (act == cur_state["prevact"] and act.type == 'U'):
                        yield (f"{act.type}:{act.label}", feats)

                scored_acts.append(ScoredAction(act, 0.0))  # logprob
            else:
                vectorizer = self.model.feat_vectorizer

                examples = FeatureSet('features',
                                      ['EXAMPLE'],
                                      None,
                                      vectorizer.transform(Counter(feats)),
                                      vectorizer)
                scores = [np.log(pred) for pred in self.model.predict(examples)[0]]

                # Convert the string labels from the classifier back into
                # ShiftReduceAction objects and sort them by their scores
                scored_acts = sorted(zip(self._get_model_actions(), scores),
                                     key=itemgetter(1),
                                     reverse=True)

            # If parsing, verify the validity of the actions.
            if gold_actions is None:
                scored_acts = [x for x in scored_acts
                               if self.is_valid_action(x[0], cur_state)]
            else:
                for x in scored_acts:
                    assert self.is_valid_action(x[0], cur_state)

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
                    queue = list(cur_state["queue"])
                    stack = list(cur_state["stack"])
                prevact = cur_state["prevact"]

                action, score = scored_acts.pop(0)

                # Add the newly created state
                tmp_state = {"prevact": action,
                             "ucnt": cur_state["ucnt"],
                             "score": cur_state["score"] + score,
                             "nsteps": cur_state["nsteps"] + 1,
                             "stack": stack,
                             "queue": queue}
                self.process_action(action, tmp_state)

                states.append(tmp_state)

        if not completetrees:
            logging.warning(f"No complete trees found. "
                            f"doc id = {doc_dict['doc_id']}")

            # Default to a flat tree if there is no complete parse.
            new_tree = Tree.fromstring("(ROOT)")
            for i in range(len(tagged_edus)):
                tmp_child = Tree.fromstring("(text)")
                tmp_child.append(i)
                new_tree.append(tmp_child)
            completetrees.append({"tree": new_tree, "score": 0.0})

        if gold_actions is None or not make_features:
            for t in completetrees:
                yield t
