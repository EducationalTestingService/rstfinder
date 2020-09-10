#!/usr/bin/env python

"""
Merge RST discourse treebank with Penn treebank to create train/test files.

This script merges the RST Discourse
Treebank (http://catalog.ldc.upenn.edu/LDC2002T07) with the Penn
Treebank (Treebank-3, http://catalog.ldc.upenn.edu/LDC99T42) and
creates JSON files for the training and test sets.

The JSON files contain lists with one dictionary per document. Each of these
dictionaries has the following keys:

- ``doc_id``: The Penn Treebank ID (e.g., wsj0764)

- ``path_basename``: the basename of the RST Discourse Treebank
  (e.g., file1.edus)

- ``tokens``: a list containing lists of tokens in the document, as extracted
  from the Peen Treebank parse trees.

- ``edu_strings``: the character strings from the RSTDTB for each elementary
  discourse unit (EDU) in the document.

- ``syntax_trees``: Syntax trees from the Penn Treebank.

- ``token_tree_positions``: A list of lists (one per sentence) of tree
  positions for the tokens in the document.  The positions are from NLTK's
  ``treeposition()`` function.

- ``pos_tags``: A list of lists (one per sentence) of part-of-speech tags.

- ``edu_start_indices``: A list of (sentence #, token #, EDU #) tuples for the
  EDUs in this document.

- ``edu_starts_paragraph``: A list of binary indicators for whether each EDU
  starts a new paragraph.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import json
import logging
import re
from glob import glob
from os.path import basename, join

import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tree import ParentedTree

from .discourse_segmentation import extract_edus_tokens
from .reformat_rst_trees import convert_parens_in_rst_tree_str, fix_rst_treebank_tree_str, reformat_rst_tree
from .tree_util import (TREE_PRINT_MARGIN,
                        convert_paren_tokens_to_ptb_format,
                        convert_ptb_tree,
                        extract_converted_terminals,
                        extract_preterminals)

# file mapping from the RSTDTB documentation
file_mapping = {"file1.edus": "wsj_0764.out.edus",
                "file2.edus": "wsj_0430.out.edus",
                "file3.edus": "wsj_0766.out.edus",
                "file4.edus": "wsj_0778.out.edus",
                "file5.edus": "wsj_2172.out.edus"}


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("rst_discourse_treebank_dir",
                        help="Directory for the RST Discourse Treebank "
                             "containing the subdirectory "
                             "`data/RSTtrees-WSJ-main-1.0`.")
    parser.add_argument("penn_treebank_dir",
                        help="Directory for the Penn Treebank containing "
                             "the subdirectory `parsed/mrg/wsj`.")
    parser.add_argument('--output_dir',
                        help="Directory where the output JSON files will "
                             "be saved.",
                        default='.')
    args = parser.parse_args()

    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'),
                        level=logging.INFO)

    logging.warning("Warnings related to minor issues that are difficult to "
                    "resolve will be logged for the following files: "
                    "file1.edus, file5.edus, wsj_0678.out.edus, and "
                    "wsj_2343.out.edus. Multiple warnings 'not enough syntax "
                    "trees' will be produced because the RSTDTB has footers "
                    "that are not in the PTB (e.g., indicating where a story "
                    "is written). Also, there are some loose match warnings "
                    "because of differences in formatting between the "
                    "treebanks.")

    # load the maxent part-of-speech tagger from NLTK which
    # will be used to tag the RST treebank sentences
    tagger = nltk.data.load("taggers/maxent_treebank_pos_tagger/PY3/english.pickle")

    # iterate over training and test partitions of the RSTDTB
    for dataset in ["TRAINING", "TEST"]:
        logging.info(dataset)
        outputs = []

        edu_paths = sorted(glob(join(args.rst_discourse_treebank_dir,
                                     "data",
                                     "RSTtrees-WSJ-main-1.0",
                                     dataset,
                                     "*.edus")))

        # iterate over each file
        for edu_path_index, edu_path in enumerate(edu_paths):

            edu_path_basename = basename(edu_path)
            tokens_doc = []
            edu_start_indices = []

            # get the corresponding PTB parse
            logging.info(f"{edu_path_index} {edu_path_basename}")
            ptb_id = (file_mapping[edu_path_basename] if
                      edu_path_basename in file_mapping else
                      edu_path_basename)[:-9]
            ptb_path = join(args.penn_treebank_dir,
                            "parsed",
                            "mrg",
                            "wsj",
                            ptb_id[4:6],
                            f"{ptb_id}.mrg")

            # normalize and convert the PTB tree
            with open(ptb_path, 'r') as ptbfh:
                doc = re.sub(r'\s+', ' ', ptbfh.read()).strip()
                trees = [ParentedTree.fromstring(f"( ({node}") for node
                         in re.split(r"\(\s*\(", doc) if node]
            for tree in trees:
                convert_ptb_tree(tree)

            # read in the RST tree and reformat
            with open(edu_path, 'r') as edufh:
                edus = [line.strip() for line in edufh.readlines()]
            edu_path_outfile = edu_path[:-5]
            dis_path = f"{edu_path_outfile}.dis"
            with open(dis_path) as disfh:
                rst_tree_str = disfh.read().strip()
                rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
                rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
                rst_tree = ParentedTree.fromstring(rst_tree_str)
                reformat_rst_tree(rst_tree)

            # identify which EDUs are at the beginnings of paragraphs
            edu_starts_paragraph = []
            with open(edu_path_outfile) as edufh:
                outfile_doc = edufh.read().strip()
                paragraphs = re.split(r'\n\n+', outfile_doc)

                # filter out paragraphs that don't include a word character
                paragraphs = [para for para in paragraphs if re.search(r'\w', para)]

                # remove extra non-word characters to make alignment easier
                # (to avoid problems with the minor discrepancies that exist
                #  in the two versions of the documents)
                paragraphs = [re.sub(r'\W', r'', para.lower())
                              for para in paragraphs]
                paragraph_idx = -1
                paragraph = ""
                for edu_index, edu in enumerate(edus):
                    logging.debug(f"edu: {edu}, "
                                  f"paragraph: {paragraph}, "
                                  f"p_idx: {paragraph_idx}")
                    edu = re.sub(r'\W', r'', edu.lower())
                    starts_paragraph = False
                    crossed_paragraphs = False
                    while len(paragraph) < len(edu):
                        assert not crossed_paragraphs or starts_paragraph
                        starts_paragraph = True
                        paragraph_idx += 1
                        paragraph += paragraphs[paragraph_idx]
                        if len(paragraph) < len(edu):
                            crossed_paragraphs = True
                            logging.warning(f"A paragraph is split across trees. "
                                            f"doc: {edu_path_basename}, "
                                            f"chars: {paragraphs[paragraph_idx:paragraph_idx + 2]}, "
                                            f"edu: {edu}")

                    assert paragraph.index(edu) == 0
                    logging.debug(f"edu_starts_paragraph = {starts_paragraph}")
                    edu_starts_paragraph.append(starts_paragraph)
                    paragraph = paragraph[len(edu):].strip()
                assert paragraph_idx == len(paragraphs) - 1
                if sum(edu_starts_paragraph) != len(paragraphs):
                    logging.warning((f"The number of sentences that start a "
                                     f"paragraph is not equal to the number "
                                     f"of paragraphs.  This is probably due "
                                     f"to trees being split across "
                                     f"paragraphs. doc: {edu_path_basename}"))

            edu_index = -1
            tok_index = 0
            tree_index = 0

            edu = ""
            tree = trees[0]
            treebank_tokenizer = TreebankWordTokenizer()

            tokens_doc = [extract_converted_terminals(tree) for tree in trees]
            tokens = tokens_doc[0]
            preterminals = [extract_preterminals(tree) for tree in trees]

            while edu_index < len(edus) - 1:
                # if we are out of tokens for the sentence we are working
                # with, move to the next sentence.
                if tok_index >= len(tokens):
                    tree_index += 1
                    if tree_index >= len(trees):
                        logging.warning((f"Not enough syntax trees "
                                         f"for {edu_path_basename}, probably "
                                         f"because the RSTDB contains a footer "
                                         f"that is not in the PTB. The remaining "
                                         f"EDUs will be automatically tagged."))

                        unparsed_edus = ' '.join(edus[edu_index + 1:])
                        # the tokenizer splits '---' into '--' '-'.
                        # this is a hack to get around that.
                        unparsed_edus = re.sub(r'---', '--', unparsed_edus)

                        # use NLTK to tag the unparsed EDUs and create trees
                        # from the tagged sentences
                        for sentence in nltk.sent_tokenize(unparsed_edus):
                            tokens = treebank_tokenizer.tokenize(sentence)
                            converted_tokens = convert_paren_tokens_to_ptb_format(tokens)
                            tagged_sent = tagger.tag(converted_tokens)
                            word_tags = " ".join([f"({tag} {word})" for word, tag
                                                  in tagged_sent])
                            new_tree = ParentedTree.fromstring(f"((S {word_tags}))")
                            trees.append(new_tree)
                            tokens_doc.append(extract_converted_terminals(new_tree))
                            preterminals.append(extract_preterminals(new_tree))

                    tree = trees[tree_index]
                    tokens = tokens_doc[tree_index]
                    tok_index = 0

                tok = tokens[tok_index]

                # if edu is the empty string, then the previous edu was
                # completed by the last token, so this token starts the
                # next edu
                if not edu:
                    edu_index += 1
                    edu = edus[edu_index]
                    edu = re.sub(r">\s*", r'', edu).replace("&amp;", '&')
                    edu = re.sub(r"---", r"--", edu)
                    edu = edu.replace(". . .", "...")

                    # annoying edge cases that we fix manually
                    if edu_path_basename == 'file1.edus':
                        edu = edu.replace('founded by',
                                          "founded by his grandfather.")
                    elif (edu_path_basename == "wsj_0660.out.edus" or
                          edu_path_basename == "wsj_1368.out.edus" or
                          edu_path_basename == "wsj_1371.out.edus"):
                        edu = edu.replace("S.p. A.", "S.p.A.")
                    elif edu_path_basename == "wsj_1329.out.edus":
                        edu = edu.replace("G.m.b. H.", "G.m.b.H.")
                    elif edu_path_basename == "wsj_1367.out.edus":
                        edu = edu.replace("-- that turban --",
                                          "-- that turban")
                    elif edu_path_basename == "wsj_1377.out.edus":
                        edu = edu.replace("Part of a Series",
                                          "Part of a Series }")
                    elif edu_path_basename == "wsj_1974.out.edus":
                        edu = edu.replace(r"5/ 16", r"5/16")
                    elif edu_path_basename == "file2.edus":
                        edu = edu.replace("read it into the record,",
                                          "read it into the record.")
                    elif edu_path_basename == "file3.edus":
                        edu = edu.replace("about $to $", "about $2 to $4")
                    elif edu_path_basename == "file5.edus":
                        # There is a PTB error in wsj_2172.mrg:
                        # The word "analysts" is missing from the parse.
                        # It's gone without a trace :-/
                        edu = edu.replace("panic among analysts",
                                          "panic among")
                        edu = edu.replace("his bid Oct. 17", "his bid Oct. 5")
                        edu = edu.replace("his bid on Oct. 17",
                                          "his bid on Oct. 5")
                        edu = edu.replace("to commit $billion,",
                                          "to commit $3 billion,")
                        edu = edu.replace("received $million in fees",
                                          "received $8 million in fees")
                        edu = edu.replace("`` in light", '"in light')
                        edu = edu.replace("3.00 a share", "2 a share")
                        edu = edu.replace(" the Deal.", " the Deal.'")
                        edu = edu.replace("' Why doesn't", "Why doesn't")
                    elif edu_path_basename == "wsj_1331.out.edus":
                        edu = edu.replace("`S", "'S")
                    elif edu_path_basename == "wsj_1373.out.edus":
                        edu = edu.replace("... An N.V.", "An N.V.")
                        edu = edu.replace("features.", "features....")
                    elif edu_path_basename == "wsj_1123.out.edus":
                        edu = edu.replace('" Reuben', 'Reuben')
                        edu = edu.replace("subscribe to.", 'subscribe to."')
                    elif edu_path_basename == "wsj_2317.out.edus":
                        edu = edu.replace(". The lower", "The lower")
                        edu = edu.replace("$4 million", "$4 million.")
                    elif edu_path_basename == 'wsj_1376.out.edus':
                        edu = edu.replace("Elizabeth.", 'Elizabeth.\'"')
                        edu = edu.replace('\'" In', "In")
                    elif edu_path_basename == "wsj_1105.out.edus":
                        # PTB error: a sentence starts with an end quote.
                        # For simplicity, we'll just make the
                        # EDU string look like the PTB sentence.
                        edu = edu.replace("By lowering prices",
                                          '"By lowering prices')
                        edu = edu.replace(' 70% off."', ' 70% off.')
                    elif edu_path_basename == 'wsj_1125.out.edus':
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace("developer.", 'developer."')
                        edu = edu.replace('"So developers', 'So developers')
                    elif edu_path_basename == "wsj_1158.out.edus":
                        edu = re.sub(r"\s*\-$", r'', edu)
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace(' virtues."', " virtues.")
                        edu = edu.replace("So much for", '"So much for')
                    elif edu_path_basename == "wsj_0632.out.edus":
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace(" individual.", ' individual."')
                        edu = edu.replace('"If there ', "If there ")
                    elif edu_path_basename == "wsj_2386.out.edus":
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace('lenders."', 'lenders.')
                        edu = edu.replace('Mr. P', '"Mr. P')
                    elif edu_path_basename == 'wsj_1128.out.edus':
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace("it down.", 'it down."')
                        edu = edu.replace('"It\'s a real"', "It's a real")
                    elif edu_path_basename == "wsj_1323.out.edus":
                        # PTB error (or at least a very unusual edge case):
                        # "--" ends a sentence.
                        edu = edu.replace("-- damn!", "damn!")
                        edu = edu.replace("from the hook", "from the hook --")
                    elif edu_path_basename == "wsj_2303.out.edus":
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace("Simpson in an interview.",
                                          'Simpson in an interview."')
                        edu = edu.replace('"Hooker\'s', 'Hooker\'s')
                    # wsj_2343.out.edus also has an error that can't be easily
                    # fixed: and EDU spans 2 sentences, ("to analyze what...").

                    if (edu_start_indices and
                            tree_index - edu_start_indices[-1][0] > 1):
                        logging.warning((f"SKIPPED A TREE. "
                                         f"file = {edu_path_basename}: "
                                         f"tree_index = {tree_index}, "
                                         f"edu_start_indices[-1][0] = {edu_start_indices[-1][0]}, "
                                         f"edu index = {edu_index}"))

                    edu_start_indices.append((tree_index, tok_index, edu_index))

                # remove the next token from the edu, along with any whitespace
                if edu.startswith(tok):
                    edu = edu[len(tok):].strip()
                elif (re.search(r"[^a-zA-Z0-9]", edu[0]) and
                      edu[1:].startswith(tok)):
                    logging.warning((f"loose match: tok = {tok}, "
                                     f"remainder of EDU: {edu}"))
                    edu = edu[len(tok) + 1:].strip()
                else:
                    m_tok = re.search(r"^[^a-zA-Z ]+$", tok)
                    m_edu = re.search(r"^[^a-zA-Z ]+(.*)", edu)
                    if not m_tok or not m_edu:
                        raise Exception((f"\n\npath_index: {edu_path_index}"
                                         f"\ntok: {tok}\n"
                                         f"edu: {edu}\n"
                                         f"full_edu:{edus[edu_index]}"
                                         f"\nleaves:{tree.leaves()}\n\n"))
                    logging.warning(f"loose match: {tok} ==> {edu}")
                    edu = m_edu.groups()[0].strip()

                tok_index += 1

            output = {"doc_id": ptb_id,
                      "path_basename": edu_path_basename,
                      "tokens": tokens_doc,
                      "edu_strings": edus,
                      "syntax_trees": [t.pformat(margin=TREE_PRINT_MARGIN)
                                       for t in trees],
                      "token_tree_positions": [[x.treeposition() for x in
                                                preterminals_sentence]
                                               for preterminals_sentence
                                               in preterminals],
                      "pos_tags": [[x.label() for x in preterminals_sentence]
                                   for preterminals_sentence in preterminals],
                      "edu_start_indices": edu_start_indices,
                      "rst_tree": rst_tree.pformat(margin=TREE_PRINT_MARGIN),
                      "edu_starts_paragraph": edu_starts_paragraph}

            assert len(edu_start_indices) == len(edus)
            assert len(edu_starts_paragraph) == len(edus)

            # check that the EDUs match up
            edu_tokens = extract_edus_tokens(edu_start_indices, tokens_doc)
            for (edu_index,
                 (edu, edu_token_list)) in enumerate(zip(edus, edu_tokens)):
                edu_nospace = re.sub(r'\s+', '', edu).lower()
                edu_tokens_nospace = ''.join(edu_token_list).lower()
                distance = nltk.distance.edit_distance(
                    edu_nospace, edu_tokens_nospace)
                if distance > 4:
                    logging.warning((f"EDIT DISTANCE > 3 IN {edu_path_basename}: "
                                     f"edu string = {edu}, "
                                     f"edu tokens = {edu_token_list}, "
                                     f"edu idx = {edu_index}"))
                if not re.search(r"[A-Za-z0-9]", edu_tokens_nospace):
                    logging.warning((f"PUNCTUATION-ONLY EDU IN "
                                     f"{edu_path_basename}: "
                                     f"edu tokens = {edu_token_list}, "
                                     f"edu idx = {edu_index}"))

            outputs.append(output)

        with open(join(args.output_dir,
                       f"rst_discourse_tb_edus_{dataset}.json"), 'w') as outfile:
            json.dump(outputs, outfile)


if __name__ == "__main__":
    main()
