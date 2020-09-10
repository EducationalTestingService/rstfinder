import re
from os.path import dirname, join, realpath

from discourseparsing.paragraph_splitting import ParagraphSplitter
from nose.tools import eq_

# TODO: replace with pathlib
MY_DIR = dirname(realpath(__file__))
PARAGRAPH_TEST_FILE = join(MY_DIR, 'test_paragraph_splitting.txt')


def test_paragraph_splitting():
    """Check that the paragraphs are split as expected."""
    splitter = ParagraphSplitter()

    # open the test file and split it into paragraphs
    with open(PARAGRAPH_TEST_FILE, 'r') as paragraphfh:
        text = paragraphfh.read()
    paragraphs = splitter.find_paragraphs(text)

    eq_(len(paragraphs), 10)

    for paragraph_num, paragraph in enumerate(paragraphs, start=1):
        assert 'paragraph {}'.format(paragraph_num) in paragraph
        eq_(paragraph[-1], '.')

        # Make sure newlines have been removed within each paragraph.
        assert '\n' not in paragraph

    # Make the total number of non-whitespace characters is the same.
    eq_(len(re.sub(r'\s', '', text)),
        len(''.join([re.sub(r'\s', '', x) for x in paragraphs])))
