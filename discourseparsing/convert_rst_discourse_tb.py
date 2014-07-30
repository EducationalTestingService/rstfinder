#!/usr/bin/env python3

'''
This script merges the RST Discourse Treebank
(http://catalog.ldc.upenn.edu/LDC2002T07) with the Penn Treebank
(Treebank-3, http://catalog.ldc.upenn.edu/LDC99T42)
and creates JSON files for the training and test sets.
The JSON files contain lists with one dictionary per document.
Each of these dictionaries has the following keys:
- ptb_id: The Penn Treebank ID (e.g., wsj0764)
- path_basename: the basename of the RST Discourse Treebank (e.g., file1.edus)
- tokens: a list of lists of tokens in the document, as extracted from the
          PTB parse trees.
- edu_strings: the character strings from the RSTDTB for each elementary
               discourse unit in the document.
- syntax_trees: Syntax trees for the Penn Treebank.
- token_tree_positions: A list of lists (one per sentence) of tree positions
                        for the tokens in the document.  The positions are
                        from NLTK's treeposition() function.
- pos_tags: A list of lists (one per sentence) of POS tags.
- edu_start_indices: A list of (sentence #, token #, EDU #) tuples for the
                     EDUs in this document.
- edu_starts_paragraph: A list of binary indicators for whether each EDU starts
                        a new paragraph.

'''

import argparse
import json
import os.path
import logging
import re
from glob import glob

from nltk.tree import ParentedTree
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer


from discourseparsing.tree_util import (convert_ptb_tree, extract_preterminals,
                                        extract_converted_terminals,
                                        TREE_PRINT_MARGIN)
from discourseparsing.reformat_rst_trees import (reformat_rst_tree,
                                                 fix_rst_treebank_tree_str,
                                                 convert_parens_in_rst_tree_str)
from discourseparsing.tree_util import convert_paren_tokens_to_ptb_format
from discourseparsing.discourse_segmentation import extract_edus_tokens


# file mapping from the RSTDTB documentation
file_mapping = {'file1.edus': 'wsj_0764.out.edus',
                'file2.edus': 'wsj_0430.out.edus',
                'file3.edus': 'wsj_0766.out.edus',
                'file4.edus': 'wsj_0778.out.edus',
                'file5.edus': 'wsj_2172.out.edus'}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rst_discourse_tb_dir',
                        help='directory for the RST Discourse Treebank.  \
                              This should have a subdirectory \
                              data/RSTtrees-WSJ-main-1.0.')
    parser.add_argument('ptb_dir',
                        help='directory for the Penn Treebank.  This should \
                              have a subdirectory parsed/mrg/wsj.')
    parser.add_argument('--output_dir',
                        help='directory where the output JSON files go.',
                        default='.')
    args = parser.parse_args()

    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)

    logging.warning(
        " Warnings related to minor issues that are difficult to resolve will" +
        " be logged for the following files: " +
        " file1.edus, file5.edus, wsj_0678.out.edus, and wsj_2343.out.edus." +
        " Multiple warnings 'not enough syntax trees' will be produced" +
        " because the RSTDTB has footers that are not in the PTB (e.g.," +
        " indicating where a story is written). Also, there are some loose" +
        " match warnings because of differences in formatting between" +
        " treebanks.")

    for dataset in ['TRAINING', 'TEST']:
        logging.info(dataset)

        outputs = []

        for path_index, path in enumerate(
                sorted(glob(os.path.join(args.rst_discourse_tb_dir,
                                         'data',
                                         'RSTtrees-WSJ-main-1.0',
                                         dataset,
                                         '*.edus')))):

            path_basename = os.path.basename(path)
            # if path_basename in file_mapping:
            #     # skip the not-so-well-formatted files "file1" to "file5"
            #     continue

            tokens_doc = []
            edu_start_indices = []

            logging.info('{} {}'.format(path_index, path_basename))
            ptb_id = (file_mapping[path_basename] if
                      path_basename in file_mapping else
                      path_basename)[:-9]
            ptb_path = os.path.join(args.ptb_dir, 'parsed', 'mrg', 'wsj',
                                    ptb_id[4:6], '{}.mrg'.format(ptb_id))

            with open(ptb_path) as f:
                doc = re.sub(r'\s+', ' ', f.read()).strip()
                trees = [ParentedTree.fromstring('( ({}'.format(x)) for x
                         in re.split(r'\(\s*\(', doc) if x]

            for t in trees:
                convert_ptb_tree(t)

            with open(path) as f:
                edus = [line.strip() for line in f.readlines()]
            path_outfile = path[:-5]
            path_dis = "{}.dis".format(path_outfile)
            with open(path_dis) as f:
                rst_tree_str = f.read().strip()
                rst_tree_str = fix_rst_treebank_tree_str(rst_tree_str)
                rst_tree_str = convert_parens_in_rst_tree_str(rst_tree_str)
                rst_tree = ParentedTree.fromstring(rst_tree_str)
                reformat_rst_tree(rst_tree)

            # Identify which EDUs are at the beginnings of paragraphs.
            edu_starts_paragraph = []
            with open(path_outfile) as f:
                outfile_doc = f.read().strip()
                paragraphs = re.split(r'\n\n+', outfile_doc)
                # Filter out any paragraphs that don't include a word character.
                paragraphs = [x for x in paragraphs if re.search(r'\w', x)]
                # Remove extra nonword characters to make alignment easier
                # (to avoid problems with the minor discrepancies that exist
                #  in the two versions of the documents.)
                paragraphs = [re.sub(r'\W', r'', p.lower())
                              for p in paragraphs]

                p_idx = -1
                paragraph = ""
                for edu_index, edu in enumerate(edus):
                    logging.debug('edu: {}, paragraph: {}, p_idx: {}'
                                  .format(edu, paragraph, p_idx))
                    edu = re.sub(r'\W', r'', edu.lower())
                    starts_paragraph = False
                    crossed_paragraphs = False
                    while len(paragraph) < len(edu):
                        assert not crossed_paragraphs or starts_paragraph
                        starts_paragraph = True
                        p_idx += 1
                        paragraph += paragraphs[p_idx]
                        if len(paragraph) < len(edu):
                            crossed_paragraphs = True
                            logging.warning(
                                'A paragraph is split across trees. paragraph' +
                                'doc: {}, chars: {}, EDU: {}'
                                .format(path_basename,
                                        paragraphs[p_idx:p_idx + 2], edu))

                    assert paragraph.index(edu) == 0
                    logging.debug('edu_starts_paragraph = {}'
                                  .format(starts_paragraph))
                    edu_starts_paragraph.append(starts_paragraph)
                    paragraph = paragraph[len(edu):].strip()
                assert p_idx == len(paragraphs) - 1
                if sum(edu_starts_paragraph) != len(paragraphs):
                    logging.warning('The number of sentences that start a ' +
                                    'paragraph is not equal to the number of ' +
                                    'paragraphs.  This is probably due to ' +
                                    'trees being split across paragraphs. ' +
                                    '  doc: {}'
                                    .format(path_basename))

            edu_index = -1
            tok_index = 0
            tree_index = 0

            edu = ""
            tree = trees[0]
            tokens_doc = [extract_converted_terminals(t) for t in trees]
            tokens = tokens_doc[0]
            preterminals = [extract_preterminals(t) for t in trees]

            while edu_index < len(edus) - 1:
                # if we are out of tokens for the sentence we are working
                # with, move to the next sentence.
                if tok_index >= len(tokens):
                    tree_index += 1
                    if tree_index >= len(trees):
                        logging.warning(('Not enough syntax trees for {}. ' +
                                         ' This is probably because the RSTDB' +
                                         ' contains a footer that is not in' +
                                         ' the PTB. The remaining EDUs will' +
                                         ' be automatically tagged.')
                                        .format(path_basename))
                        unparsed_edus = ' '.join(edus[edu_index + 1:])
                        # The tokenizer splits '---' into '--' '-'.
                        # This is a hack to get around that.
                        unparsed_edus = re.sub(r'---', '--', unparsed_edus)
                        for tagged_sent in \
                            [nltk.pos_tag(convert_paren_tokens_to_ptb_format( \
                             TreebankWordTokenizer().tokenize(x)))
                             for x in nltk.sent_tokenize(unparsed_edus)]:
                            new_tree = ParentedTree.fromstring('((S {}))' \
                                .format(' '.join(['({} {})'.format(tag, word)
                                                  for word, tag
                                                  in tagged_sent])))
                            trees.append(new_tree)
                            tokens_doc.append(
                                extract_converted_terminals(new_tree))
                            preterminals.append(extract_preterminals(new_tree))

                    tree = trees[tree_index]
                    tokens = tokens_doc[tree_index]
                    tok_index = 0

                tok = tokens[tok_index]

                # if edu is the empty string, then the previous edu was
                # completed by the last token,
                # so this token starts the next edu.
                if not edu:
                    edu_index += 1
                    edu = edus[edu_index]
                    edu = re.sub(r'>\s*', r'', edu).replace('&amp;', '&')
                    edu = re.sub(r'---', r'--', edu)
                    edu = edu.replace('. . .', '...')

                    # annoying edge cases
                    if path_basename == 'file1.edus':
                        edu = edu.replace('founded by',
                                          'founded by his grandfather.')
                    elif (path_basename == 'wsj_0660.out.edus'
                          or path_basename == 'wsj_1368.out.edus'
                          or path_basename == "wsj_1371.out.edus"):
                        edu = edu.replace('S.p. A.', 'S.p.A.')
                    elif path_basename == 'wsj_1329.out.edus':
                        edu = edu.replace('G.m.b. H.', 'G.m.b.H.')
                    elif path_basename == 'wsj_1367.out.edus':
                        edu = edu.replace('-- that turban --',
                                          '-- that turban')
                    elif path_basename == 'wsj_1377.out.edus':
                        edu = edu.replace('Part of a Series',
                                          'Part of a Series }')
                    elif path_basename == 'wsj_1974.out.edus':
                        edu = edu.replace(r'5/ 16', r'5/16')
                    elif path_basename == 'file2.edus':
                        edu = edu.replace('read it into the record,',
                                          'read it into the record.')
                    elif path_basename == 'file3.edus':
                        edu = edu.replace('about $to $', 'about $2 to $4')
                    elif path_basename == 'file5.edus':
                        # There is a PTB error in wsj_2172.mrg:
                        # The word "analysts" is missing from the parse.
                        # It's gone without a trace :-/
                        edu = edu.replace('panic among analysts',
                                          'panic among')
                        edu = edu.replace('his bid Oct. 17', 'his bid Oct. 5')
                        edu = edu.replace('his bid on Oct. 17',
                                          'his bid on Oct. 5')
                        edu = edu.replace('to commit $billion,',
                                          'to commit $3 billion,')
                        edu = edu.replace('received $million in fees',
                                          'received $8 million in fees')
                        edu = edu.replace('`` in light', '"in light')
                        edu = edu.replace('3.00 a share', '2 a share')
                        edu = edu.replace(" the Deal.", " the Deal.'")
                        edu = edu.replace("' Why doesn't", "Why doesn't")
                    elif path_basename == 'wsj_1331.out.edus':
                        edu = edu.replace('`S', "'S")
                    elif path_basename == 'wsj_1373.out.edus':
                        edu = edu.replace('... An N.V.', 'An N.V.')
                        edu = edu.replace('features.', 'features....')
                    elif path_basename == 'wsj_1123.out.edus':
                        edu = edu.replace('" Reuben', 'Reuben')
                        edu = edu.replace('subscribe to.', 'subscribe to."')
                    elif path_basename == 'wsj_2317.out.edus':
                        edu = edu.replace('. The lower', 'The lower')
                        edu = edu.replace('$4 million', '$4 million.')
                    elif path_basename == 'wsj_1376.out.edus':
                        edu = edu.replace('Elizabeth.', 'Elizabeth.\'"')
                        edu = edu.replace('\'" In', 'In')
                    elif path_basename == 'wsj_1105.out.edus':
                        # PTB error: a sentence starts with an end quote.
                        # For simplicity, we'll just make the
                        # EDU string look like the PTB sentence.
                        edu = edu.replace('By lowering prices',
                                          '"By lowering prices')
                        edu = edu.replace(' 70% off."', ' 70% off.')
                    elif path_basename == 'wsj_1125.out.edus':
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace('developer.', 'developer."')
                        edu = edu.replace('"So developers', 'So developers')
                    elif path_basename == 'wsj_1158.out.edus':
                        edu = re.sub(r'\s*\-$', r'', edu)
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace(' virtues."', ' virtues.')
                        edu = edu.replace('So much for', '"So much for')
                    elif path_basename == 'wsj_0632.out.edus':
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace(' individual.', ' individual."')
                        edu = edu.replace('"If there ', 'If there ')
                    elif path_basename == 'wsj_2386.out.edus':
                        # PTB error: a sentence starts with an end quote.
                        edu = edu.replace('lenders."', 'lenders.')
                        edu = edu.replace('Mr. P', '"Mr. P')
                    elif path_basename == 'wsj_1128.out.edus':
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace('it down.', 'it down."')
                        edu = edu.replace('"It\'s a real"', "It's a real")
                    elif path_basename == 'wsj_1323.out.edus':
                        # PTB error (or at least a very unusual edge case):
                        # "--" ends a sentence.
                        edu = edu.replace('-- damn!', 'damn!')
                        edu = edu.replace('from the hook', 'from the hook --')
                    elif path_basename == 'wsj_2303.out.edus':
                        # PTB error: a sentence ends with an start quote.
                        edu = edu.replace('Simpson in an interview.',
                                          'Simpson in an interview."')
                        edu = edu.replace('"Hooker\'s', 'Hooker\'s')
                    # wsj_2343.out.edus also has an error that can't be easily
                    # fixed: and EDU spans 2 sentences, ("to analyze what...").

                    if edu_start_indices \
                            and tree_index - edu_start_indices[-1][0] > 1:
                        logging.warning(("SKIPPED A TREE. file = {}" +
                                         " tree_index = {}," +
                                         " edu_start_indices[-1][0] = {}," +
                                         " edu index = {}")
                                        .format(path_basename, tree_index,
                                                edu_start_indices[-1][0],
                                                edu_index))

                    edu_start_indices.append((tree_index, tok_index,
                                              edu_index))

                # remove the next token from the edu, along with any whitespace
                if edu.startswith(tok):
                    edu = edu[len(tok):].strip()
                elif (re.search(r'[^a-zA-Z0-9]', edu[0])
                      and edu[1:].startswith(tok)):
                    logging.warning(("loose match: tok = {}, " +
                                     "remainder of EDU: {}").format(tok, edu))
                    edu = edu[len(tok) + 1:].strip()
                else:
                    m_tok = re.search(r'^[^a-zA-Z ]+$', tok)
                    m_edu = re.search(r'^[^a-zA-Z ]+(.*)', edu)
                    if not m_tok or not m_edu:
                        raise Exception(('\n\npath_index: {}\ntok: {}\n' +
                                         'edu: {}\nfull_edu:{}\nleaves:' +
                                         '{}\n\n').format(path_index, tok, edu,
                                                          edus[edu_index],
                                                          tree.leaves()))
                    logging.warning("loose match: {} ==> {}".format(tok, edu))
                    edu = m_edu.groups()[0].strip()

                tok_index += 1

            output = {"ptb_id": ptb_id,
                      "path_basename": path_basename,
                      "tokens": tokens_doc,
                      "edu_strings": edus,
                      "syntax_trees": [t.pprint(margin=TREE_PRINT_MARGIN)
                                       for t in trees],
                      "token_tree_positions": [[x.treeposition() for x in
                                                preterminals_sentence]
                                               for preterminals_sentence
                                               in preterminals],
                      "pos_tags": [[x.label() for x in preterminals_sentence]
                                   for preterminals_sentence in preterminals],
                      "edu_start_indices": edu_start_indices,
                      "rst_tree": rst_tree.pprint(margin=TREE_PRINT_MARGIN),
                      "edu_starts_paragraph": edu_starts_paragraph}

            assert len(edu_start_indices) == len(edus)
            assert len(edu_starts_paragraph) == len(edus)

            # check that the EDUs match up
            edu_tokens = extract_edus_tokens(edu_start_indices, tokens_doc)
            for edu_index, (edu, edu_token_list) \
                    in enumerate(zip(edus, edu_tokens)):
                edu_nospace = re.sub(r'\s+', '', edu).lower()
                edu_tokens_nospace = ''.join(edu_token_list).lower()
                distance = nltk.metrics.distance.edit_distance(
                    edu_nospace, edu_tokens_nospace)
                if distance > 4:
                    logging.warning(("EDIT DISTANCE > 3 IN {}: " +
                                     "edu string = {}, edu tokens = {}, " +
                                     "edu idx = {}")
                                    .format(path_basename, edu,
                                            edu_token_list, edu_index))
                if not re.search(r'[A-Za-z0-9]', edu_tokens_nospace):
                    logging.warning(("PUNCTUATION-ONLY EDU IN {}: " +
                                     "edu tokens = {}, edu idx = {}")
                                    .format(path_basename, edu_token_list,
                                            edu_index))

            outputs.append(output)

        with open(os.path.join(args.output_dir, ('rst_discourse_tb_edus_' +
                                                 '{}.json').format(dataset)),
                  'w') as outfile:
            json.dump(outputs, outfile)


if __name__ == '__main__':
    main()
