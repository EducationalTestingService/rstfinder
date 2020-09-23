import re
from pathlib import Path

from nose.tools import eq_
from rstfinder.paragraph_splitting import ParagraphSplitter

MY_DIR = Path(__file__).parent


def test_paragraph_splitting():
    """Check that the paragraphs are split as expected."""
    splitter = ParagraphSplitter()

    # open the test file and split it into paragraphs
    with open(MY_DIR / "data" / "paragraphs.txt", 'r') as paragraphfh:
        text = paragraphfh.read()
    paragraphs = splitter.find_paragraphs(text)

    eq_(len(paragraphs), 10)

    for paragraph_num, paragraph in enumerate(paragraphs, start=1):
        assert f"paragraph {paragraph_num}" in paragraph
        eq_(paragraph[-1], '.')

        # Make sure newlines have been removed within each paragraph.
        assert '\n' not in paragraph

    # Make the total number of non-whitespace characters is the same.
    eq_(len(re.sub(r'\s', '', text)),
        len(''.join([re.sub(r'\s', '', x) for x in paragraphs])))
