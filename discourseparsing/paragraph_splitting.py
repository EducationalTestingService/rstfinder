#!/usr/bin/env python3

import logging
import re

from discourseparsing.io_util import read_text_file


class ParagraphSplitter(object):
    '''
    Defines a paragraph to be contiguous text starting and ending with a
    non-whitespace character that does not include a sequence of a newline
    followed by a) another newline, b) 3 or more spaces, or c) a tab.

    Note: carriage returns are ignored.  Sorry, Mac OS 9 users.  :-/
    '''

    def __init__(self):
        pass

    @staticmethod
    def find_paragraphs(text):
        # Remove carriage returns and leading/trailing whitespace.
        text = re.sub(r'\r', r'', text.strip())

        # Note that ":?" makes the parenthesized thing not count as a group.
        res = re.split(r'\n\s*(?:\n|\s\s\s|\t)\s*', text)

        # If we only found one long paragraph, try splitting by newlines.
        if len(res) == 1 and len(text) > 500 and re.search(r'\.\s*\n', text):
            logging.info('The text was over 500 characters, no indentation' +
                         ' or blank lines were found, and there is a period' +
                         ' followed by a newline. Falling back to splitting' +
                         ' by newlines.')
            res = re.split(r'\n+', text)

        # Replace multiple spaces/newlines within a paragraph with one space.
        res = [re.sub(r'\s+', ' ', x) for x in res]

        # Make sure the number of non-whitespace characters is unchanged.
        assert len(re.sub(r'\s', '', text)) \
            == len(''.join([re.sub(r'\s', '', x) for x in res]))

        logging.info('Number of paragraphs found: {}'.format(len(res)))

        return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_text', help='raw text to split into paragraphs')
    args = parser.parse_args()

    doc = read_text_file(args.input_text)
    paragraphs = ParagraphSplitter.find_paragraphs(doc)
    for paragraph in paragraphs:
        print(paragraph)
