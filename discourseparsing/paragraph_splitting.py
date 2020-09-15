#!/usr/bin/env python
"""
Split given text into paragraphs.

This script takes in a document and splits it into paragraphs where a
paragraph is defined as follows: contiguous text starting and ending with
a non-whitespace character that does not include a sequence of a newline
followed by a) another newline, b) 3 or more spaces, or c) a tab.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse
import logging
import re

from .io_util import read_text_file


class ParagraphSplitter(object):
    """
    A class containing the paragraph splitter.

    Defines a paragraph to be contiguous text starting and ending with a
    non-whitespace character that does not include a sequence of a newline
    followed by a) another newline, b) 3 or more spaces, or c) a tab.

    Note: carriage returns are ignored.
    """

    @staticmethod
    def find_paragraphs(text, doc_id=None):
        """
        Find paragraphs in the given document.

        Parameters
        ----------
        text : str
            The input document in which to find paragraphs.
        doc_id : str, optional
            An optional ID for the given document.

        Returns
        -------
        res : list
            The list of found paragraphs in the document.
        """
        # remove carriage returns and leading/trailing whitespace
        text = re.sub(r'\r', r'', text.strip())

        # note that ":?" makes the parenthesized thing not count as a group
        res = re.split(r"\n\s*(?:\n|\s\s\s|\t)\s*", text)

        # if we only found one long paragraph, try splitting by newlines
        if len(res) == 1 and len(text) > 500 and re.search(r'\.\s*\n', text):
            logging.info(f"The text was over 500 characters, no indentation "
                         f"or blank lines were found, and there is a period "
                         f"followed by a newline. Falling back to splitting "
                         f"by newlines. doc_id = {doc_id}")
            res = re.split(r"\n+", text)

        # replace multiple spaces/newlines within a paragraph with one space
        res = [re.sub(r"\s+", ' ', x) for x in res]

        # make sure the number of non-whitespace characters is unchanged
        original_characters = len(re.sub(r'\s', '', text))
        result_characters = len(''.join([re.sub(r'\s', '', x) for x in res]))
        assert original_characters == result_characters

        logging.info(f"Number of paragraphs found: {len(res)}")

        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_text",
                        help="Input text to split into paragraphs")
    args = parser.parse_args()

    # read the contents of the file
    doc = read_text_file(args.input_text)

    # split into paragraphs and print out each paragraph
    paragraphs = ParagraphSplitter.find_paragraphs(doc, args.input_text)
    for paragraph in paragraphs:
        print(paragraph)
