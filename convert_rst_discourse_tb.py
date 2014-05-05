#!/usr/bin/env python3

'''
This script merges the RST Discourse Treebank
(http://catalog.ldc.upenn.edu/LDC2002T07) with the Penn Treebank
(Treebank-3, http://catalog.ldc.upenn.edu/LDC99T42)
and creates JSON files for the training and test sets.
The JSON files contain lists with one dictionary per document.
Each of these dictionaries has the following keys:
- ptb_id: The Penn Treebank ID (e.g., wsj0764)
- path_basename: the basename of the RST Discourse Treebank (e.g., file1.out)
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

'''

from glob import glob
import os.path
import re
import sys
import json
import argparse
#import warnings

#from pyparsing import OneOrMore, nestedExpr
from nltk.tree import ParentedTree

# file mapping from the RSTDTB documentation
file_mapping = {'file1.edus': 'wsj_0764.out.edus',
                'file2.edus': 'wsj_0430.out.edus',
                'file3.edus': 'wsj_0766.out.edus',
                'file4.edus': 'wsj_0778.out.edus',
                'file5.edus': 'wsj_2172.out.edus',}

def convert_ptb_tree(t):
    # Remove traces, etc.
    for subtree in [x for x in
                    t.subtrees(filter=lambda x: x.label() == '-NONE-')]:
        curtree = subtree
        while curtree.label() == '-NONE-' or len(curtree) == 0:
            parent = curtree.parent()
            parent.remove(curtree)
            curtree = parent

    # Remove suffixes that don't appear in typical parser output
    # (e.g., "-SBJ-1" in "NP-SBJ-1").
    # Leave labels starting with "-" as is (e.g., "-LRB-").
    for subtree in t.subtrees():
        label = subtree.label()
        if '-' in label and label[0] != '-':
            subtree.set_label(label[:label.index('-')])
        label = subtree.label()
        if '=' in label and label[0] != '=':
            subtree.set_label(label[:label.index('=')])


def extract_converted_terminals(tree):
    res = []
    prev_w = ""
    for w in tree.leaves():
        if prev_w and prev_w == "U.S." and w == ".":
            continue
        if w == '-LCB-':
            w = '{'
        elif w == '-RCB-':
            w = '}'
        elif w == '-LRB-':
            w = '('
        elif w == '-RRB-':
            w = ')'
        elif w == '``' or w == "''":
            w = '"'

        w = re.sub(r'\\', r'', w)
        prev_w = w
        res.append(w)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rst_discourse_tb_dir', help='directory for the RST Discourse Treebank.  This should have a subdirectory data/RSTtrees-WSJ-main-1.0.')
    parser.add_argument('ptb_dir', help='directory for the Penn Treebank.  This should have a subdirectory parsed/mrg/wsj.')
    args = parser.parse_args()

    #rst_discourse_tb_dir = '/Users/mheilman/corpora/rst_discourse_treebank'
    output_dir = '.'
    outputs = []

    for dataset in ['TRAINING', 'TEST']:
        print(dataset, file=sys.stderr)

        for path_index, path in enumerate(sorted(glob(os.path.join(args.rst_discourse_tb_dir, 'data', 'RSTtrees-WSJ-main-1.0', dataset, '*.edus')))):
            tokens_doc = []
            edu_start_indices = []

            path_basename = os.path.basename(path)
            print('{} {}'.format(path_index, path_basename), file=sys.stderr)
            ptb_id = (file_mapping[path_basename] if path_basename in file_mapping else path_basename)[:-9]
            ptb_path = os.path.join(args.ptb_dir, 'parsed', 'mrg', 'wsj',
                                    ptb_id[4:6], '{}.mrg'.format(ptb_id))

            with open(ptb_path) as f:
                doc = re.sub(r'\s+', ' ', f.read()).strip()
                trees = [ParentedTree('( ({}'.format(x)) for x
                         in re.split(r'\(\s*\(', doc) if x]

            for tree in trees:
                convert_ptb_tree(tree)

            with open(path) as f:
                edus = [line.strip() for line in f.readlines()]
            path_dis = "{}.dis".format(path[:-5])
            with open(path_dis) as f:
                #warnings.filterwarnings("ignore", category=DeprecationWarning)
                #rst_tree = OneOrMore(nestedExpr()).parseString(f.read().strip()).asList()
                #warnings.filterwarnings("always", category=DeprecationWarning)
                rst_tree = f.read().strip()

            edu_index = -1
            tok_index = 0
            tree_index = 0

            edu = ""
            tree = trees[0]
            tokens_doc = [extract_converted_terminals(tree) for tree in trees]
            tokens = tokens_doc[0]
            preterminals = [[node for node in tree.subtrees()
                             if isinstance(node[0], str)]
                            for tree in trees]

            while edu_index < len(edus) - 1:
                # if we are out of tokens for the sentence we are working
                # with, move to the next sentence.
                if tok_index >= len(tokens):
                    tree_index += 1
                    if tree_index >= len(trees):
                        break
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
                    if (path_basename == 'wsj_0660.out.edus'
                        or path_basename == 'wsj_1368.out.edus'
                        or path_basename == "wsj_1371.out.edus"):
                        edu = edu.replace('S.p. A.', 'S.p.A.')
                    elif path_basename == 'wsj_1329.out.edus':
                        edu = edu.replace('G.m.b. H.', 'G.m.b.H.')
                    elif path_basename == 'wsj_1158.out.edus':
                        edu = re.sub(r'\s*\-$', r'', edu)
                    elif path_basename == 'wsj_1367.out.edus':
                        edu = edu.replace('-- that turban --', '-- that turban')
                    elif path_basename == 'wsj_1377.out.edus':
                        edu = edu.replace(r'Part of a Series', r'Part of a Series }')
                        edu = edu.replace(r'(All buyers 47%)', r'')
                    elif path_basename == 'wsj_1974.out.edus':
                        edu = edu.replace(r'5/ 16', r'5/16')
                    elif path_basename == 'file2.edus':
                        edu = edu.replace('read it into the record,', 'read it into the record.')
                    elif path_basename == 'file3.edus':
                        edu = edu.replace('about $to $', 'about $2 to $4')
                    elif path_basename == 'file5.edus':
                        edu = edu.replace('panic among analysts', 'panic among')
                        edu = edu.replace('his bid Oct. 17', 'his bid Oct. 5')
                        edu = edu.replace('his bid on Oct. 17', 'his bid on Oct. 5')
                        edu = edu.replace('to commit $billion,', 'to commit $3 billion,')
                        edu = edu.replace('received $million in fees', 'received $8 million in fees')
                        edu = edu.replace('`` in light', '"in light')
                        edu = edu.replace('3.00 a share', '2 a share')
                    elif path_basename == 'wsj_1331.out.edus':
                        edu = edu.replace('`S', "'S")
                    elif path_basename == 'wsj_1373.out.edus':
                        edu = edu.replace('.. An N.V.', 'An N.V.')

                    edu_start_indices.append((tree_index, tok_index, edu_index))

                # remove the next token from the edu, along with any whitespace
                if edu.startswith(tok):
                    edu = edu[len(tok):].strip()
                elif re.search(r'[^a-zA-Z0-9]', edu[0]) and edu[1:].startswith(tok):
                    print("loose match: {} {}".format(tok, edu),
                          file=sys.stderr)
                    edu = edu[len(tok) + 1:].strip()
                else:
                    m_tok = re.search(r'^[^a-zA-Z ]+$', tok)
                    m_edu = re.search(r'^[^a-zA-Z ]+(.*)', edu)
                    if not m_tok or not m_edu:
                        raise Exception('\n\npath_index: {}\ntok: {}\nedu: {}\nfull_edu:{}\nleaves:{}\n\n'.format(path_index, tok, edu, edus[edu_index], tree.leaves()))
                    print("loose match: {} ==> {}".format(tok, edu),
                          file=sys.stderr)
                    edu = m_edu.groups()[0].strip()

                tok_index += 1

            output = {"ptb_id": ptb_id,
                      "path_basename": path_basename,
                      "tokens": tokens_doc,
                      "edu_strings": edus,
                      "syntax_trees": [t.pprint() for t in trees],
                      "token_tree_positions": [[x.treeposition() for x in
                                                preterminals_sentence]
                                               for preterminals_sentence
                                               in preterminals],
                      "pos_tags": [[x.label() for x in preterminals_sentence]
                                   for preterminals_sentence in preterminals],
                      "edu_start_indices": edu_start_indices,
                      "rst_tree": rst_tree}
            outputs.append(output)

        with open(os.path.join(output_dir, 'rst_discourse_tb_edus_{}.json'.format(dataset)), 'w') as outfile:
            json.dump(outputs, outfile)


if __name__ == '__main__':
    main()
