#!/usr/bin/env python

"""
Segment and RST-parse a document.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import codecs
import json
import logging
import re

from nltk.tree import ParentedTree

from .discourse_parsing import Parser
from .discourse_segmentation import Segmenter, extract_edus_tokens
from .io_util import read_text_file
from .parse_util import SyntaxParserWrapper
from .tree_util import TREE_PRINT_MARGIN, extract_converted_terminals, extract_preterminals


def segment_and_parse(doc_dict, syntax_parser, segmenter, rst_parser):
    """
    Syntax parse, segment, and RST parse the given document, as necessary.

    This function performs syntax parsing, discourse segmentation, and RST
    parsing as necessary, given a (partial) document dictionary. See
    ``convert_rst_discourse_tb.py`` for more about document dictionaries.

    The only required fields in the document dictionary (``doc_dict``) are
    "doc_id" and "raw_text". In addition to returning the EDU tokens and the
    RST trees, this function also modifies the given document dictionary in
    place to add other fields with the outputs of the syntactic parser,
    the segmenter, and the RST parser.

    Note that the document dictionary can be partially populated in order
    to skip some of the processing:

    - If the fields "syntax_trees" is present in the dictionary, then
      the function assumes that constituency parsing is not required
      and that the following additional fields will also be present:
      "starts_paragraph_list", "tokens", "token_tree_positions", and
      "pos_tags".

    - If the field "edu_start_indices" is present in the dictionary, then
      the function assumes that discourse segmentation is not required and
      that the following additional field will also be present:
      "edu_starts_paragraph".

    Parameters
    ----------
    doc_dict : dict
        A dictionary representing the document.
    syntax_parser : parse_util.SyntaxParserWrapper
         An instance of the syntactic parser wrapper.
    segmenter : discourse_segmentation.Segmenter
        An instance of the discourse segmenter.
    rst_parser : discourse_parsing.Parser
        An instance of the RST parser.

    Returns
    -------
    results : tuple
        A 2-tuple containing two lists: one for the EDU tokens and one for the
        RST trees.
    """
    # return empty lists if the input was blank; check whether raw_text is
    # available so this does not crash when evaluating on pre-parsed
    # treebank documents
    if "raw_text" in doc_dict and not doc_dict["raw_text"].strip():
        logging.warning(f"The input contained no non-whitespace characters."
                        f" doc_id = {doc_dict['doc_id']}")
        return [], []

    if "syntax_trees" not in doc_dict:
        # do syntactic parsing
        trees, starts_paragraph_list = syntax_parser.parse_document(doc_dict)
        doc_dict["syntax_trees"] = [tree.pformat(margin=TREE_PRINT_MARGIN)
                                    for tree in trees]
        doc_dict["starts_paragraph_list"] = starts_paragraph_list
        preterminals = [extract_preterminals(tree) for tree in trees]
        doc_dict["token_tree_positions"] = [[x.treeposition() for x in
                                             preterminals_sentence]
                                            for preterminals_sentence
                                            in preterminals]
        doc_dict["tokens"] = [extract_converted_terminals(tree) for tree in trees]
        doc_dict["pos_tags"] = [[x.label() for x in preterminals_sentence]
                                for preterminals_sentence in preterminals]

    if "edu_start_indices" not in doc_dict:
        # do discourse segmentation
        segmenter.segment_document(doc_dict)

        # extract whether each EDU starts a paragraph
        edu_starts_paragraph = []
        for tree_idx, tok_idx, _ in doc_dict["edu_start_indices"]:
            val = (tok_idx == 0 and doc_dict["starts_paragraph_list"][tree_idx])
            edu_starts_paragraph.append(val)
        assert len(edu_starts_paragraph) == len(doc_dict["edu_start_indices"])
        doc_dict["edu_starts_paragraph"] = edu_starts_paragraph

    # extract a list of lists of (word, POS) tuples
    edu_tokens = extract_edus_tokens(doc_dict["edu_start_indices"],
                                     doc_dict["tokens"])

    # do RST parsing
    rst_parse_trees = rst_parser.parse(doc_dict)

    return edu_tokens, rst_parse_trees


def from_constituency_trees(tree_strings, segmenter, rst_parser):
    """
    Segment and RST parse the document as represented by its constituency trees.

    This function performs discourse segmentation and RST parsing, given a list
    of constituency trees as strings. All other necessary prerequisites for
    discourse segmentation and discourse parsing are computed from the given
    constituency trees.

    The trees _must_ be computed from strings that have the same bracket
    normalization applied as the texts in the Penn Treebank, i.e., '(' and ')
    should have been converted to' "-LRB-" and "-RRB-" respectively, '[' and ']'
    to '-LSB-' and "-RSB-" respectively, and '{' and '}' to "-LCB-" and "-RCB-"
    respectively. Any paragraph boundaries must be indicated as empty strings
    and included in the list. For example, if your original document looks like
    this:

        S1 S2 S3

        S4 S5

        S6 S7 S8 S9

    where "S" represents a sentence, then you have 3 paragraphs, and the list
    of tree strings should look like this:

        [P1, P2 P3, "", P4, P5, "", P6, P7, P8, P9]

    where "P<N>" represents the parse tree for the sentence "S<N>".

    If there are no empty strings, all sentences will be assumed to be in a
    _single_ paragraph.

    Parameters
    ----------
    tree_strings : list of str
        List of strings. Each item represents a tree and should be a
        valid bracketed tree string and PTB-normalized.
    segmenter : discourse_segmentation.Segmenter
        An instance of the discourse segmenter.
    rst_parser : discourse_parsing.Parser
        An instance of the RST parser.

    Returns
    -------
    results : tuple
        A 3-tuple containing:
        (1) the document dictionary that was internally created from the
            constituency trees
        (2) a list containing the EDU tokens.
        (3) a list containing the RST trees

    Raises
    ------
    ValueError
        If any of the tree strings are incorrectly formatted.
    """
    # return empty output if the list of trees is empty
    if len(tree_strings) == 0:
        logging.warning("The input contained no trees.")
        return [], [], []

    # first convert each tree into a `ParentedTree` instance and figure
    # out which tree corresponds to a sentence that starts a new paragraph
    # in the document; paragraph boundaries will be represented by empty
    # strings in the list of trees
    trees = []
    starts_paragraph_list = []
    previous_tree_string = ""
    for tree_string in tree_strings:
        if len(tree_string) > 0:
            try:
                tree = ParentedTree.fromstring(tree_string)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid format: {tree_string}. Please check "
                                 f"that the tree is correctly formatted.")
                return [], []
            else:
                trees.append(tree)

                # the current tree starts a paragraph if the previous tree
                # was the empty string indicating a paragraph boundary
                starts_paragraph = True if len(previous_tree_string) == 0 else False
                starts_paragraph_list.append(starts_paragraph)
        previous_tree_string = tree_string

    # next compute all of the other required prerequisites from the parse trees
    preterminals = [extract_preterminals(tree) for tree in trees]
    token_tree_positions = [[x.treeposition() for x in preterminals_sentence]
                            for preterminals_sentence in preterminals]
    tokens = [extract_converted_terminals(tree) for tree in trees]
    raw_text = "\n".join([" ".join(token_list) for token_list in tokens])
    pos_tags = [[x.label() for x in preterminals_sentence]
                for preterminals_sentence in preterminals]

    # create the document dictionary with all required fields
    doc_dict = {"doc_id": "document",
                "raw_text": raw_text,
                "syntax_trees": [tree.pformat(margin=TREE_PRINT_MARGIN)
                                 for tree in trees],
                "starts_paragraph_list": starts_paragraph_list,
                "token_tree_positions": token_tree_positions,
                "tokens": tokens,
                "pos_tags": pos_tags}

    # first do discourse segmentation
    segmenter.segment_document(doc_dict)

    # extract whether each EDU starts a paragraph
    edu_starts_paragraph = []
    for tree_idx, tok_idx, _ in doc_dict["edu_start_indices"]:
        val = (tok_idx == 0 and doc_dict["starts_paragraph_list"][tree_idx])
        edu_starts_paragraph.append(val)
    assert len(edu_starts_paragraph) == len(doc_dict["edu_start_indices"])
    doc_dict["edu_starts_paragraph"] = edu_starts_paragraph

    # next extract a list of lists of (word, POS) tuples
    edu_tokens = extract_edus_tokens(doc_dict["edu_start_indices"],
                                     doc_dict["tokens"])

    # finally, do RST parsing
    rst_parse_trees = rst_parser.parse(doc_dict)

    return doc_dict, edu_tokens, rst_parse_trees


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_paths",
                        nargs='+',
                        help="Document(s) to segment and parse. Paragraphs "
                             "should be separated by two or more newline "
                             "characters.")
    parser.add_argument("-g",
                        "--segmentation_model",
                        help="Path to segmentation model.",
                        required=True)
    parser.add_argument("-p",
                        "--parsing_model",
                        help="Path to RST parsing model.",
                        required=True)
    parser.add_argument("-a",
                        "--max_acts",
                        help="Maximum number of actions for the RST parser",
                        type=int,
                        default=1)
    parser.add_argument("-n",
                        "--n_best",
                        help="Number of RST parses to return",
                        type=int,
                        default=1)
    parser.add_argument("-s",
                        "--max_states",
                        help="Maximum number of states to retain for "
                             "best-first search",
                        type=int,
                        default=1)
    parser.add_argument("-zp",
                        "--zpar_port",
                        required=False,
                        type=int)
    parser.add_argument("-zh",
                        "--zpar_hostname",
                        required=False,
                        default=None)
    parser.add_argument("-zm",
                        "--zpar_model_directory",
                        default=None)
    parser.add_argument("-v",
                        "--verbose",
                        help="Print more status information. For every "
                             "additional time this flag is specified, "
                             "output gets more verbose.",
                        default=0,
                        action="count")
    args = parser.parse_args()

    # convert verbose flag to logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]

    # format warnings more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=("%(asctime)s - %(name)s - %(levelname)s - "
                                "%(message)s"),
                        level=log_level)

    # load the various models
    logging.info("Loading models")
    syntax_parser = SyntaxParserWrapper(port=args.zpar_port,
                                        hostname=args.zpar_hostname,
                                        zpar_model_directory=args.zpar_model_directory)
    segmenter = Segmenter(args.segmentation_model)

    parser = Parser(max_acts=args.max_acts,
                    max_states=args.max_states,
                    n_best=args.n_best)
    parser.load_model(args.parsing_model)

    # iterate over the given documents
    for input_path in args.input_paths:
        logging.info(f"parsing input file: {input_path}")
        doc = read_text_file(input_path)
        logging.debug(f"doc_id = {input_path}, text = {doc}")
        doc_dict = {"raw_text": doc, "doc_id": input_path}

        edu_tokens, complete_trees = segment_and_parse(doc_dict,
                                                       syntax_parser,
                                                       segmenter,
                                                       parser)

        # we need to convert ``complete_trees`` to a list since we need
        # to iterate over this twice
        complete_trees = list(complete_trees)

        print(json.dumps({"edu_tokens": edu_tokens,
                          "scored_rst_trees":
                          [{"score": tree["score"],
                            "tree": tree["tree"].pformat(margin=TREE_PRINT_MARGIN)}
                           for tree in complete_trees]}))

        for i, tree in enumerate(complete_trees, start=1):
            ptree_str = tree["tree"].__repr__() + "\n"
            with codecs.open(f"{input_path}_{str(i)}.parentedtree", 'w', "utf-8") as ptree_file:
                ptree_file.write(ptree_str)


if __name__ == "__main__":
    main()
