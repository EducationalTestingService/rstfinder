#!/usr/bin/env python

import logging
import os
import re

from discourseparsing.paragraph_splitting import ParagraphSplitter


def test_paragraph_splitting():
    splitter = ParagraphSplitter()

    my_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(my_dir, 'test_paragraph_splitting.txt')) as f:
        text = f.read()

    paragraphs = splitter.find_paragraphs(text)

    assert len(paragraphs) == 10

    for i, paragraph in enumerate(paragraphs):
        assert 'paragraph {}'.format(i + 1) in paragraph
        assert paragraph[-1] == "."

        # Make sure newlines have been removed within each paragraph.
        assert '\n' not in paragraph

    # Make the total number of non-whitespace characters is the same.
    assert len(re.sub(r'\s', '', text)) \
        == len(''.join([re.sub(r'\s', '', x) for x in paragraphs]))


if __name__ == '__main__':
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=logging.INFO)
    test_paragraph_splitting()
    print("If no assertions failed, then this passed.")
