import os
from pathlib import Path

from nltk.tree import ParentedTree
from nose import SkipTest
from nose.tools import eq_
from rstfinder.parse_util import SyntaxParserWrapper

MY_DIR = Path(__file__).parent


def test_syntax_wrapper():
    """Test the syntactic parser wrapper."""
    # make sure the relevant environment variables are set
    if not os.getenv("ZPAR_MODEL_DIR"):
        raise SkipTest("ZPAR_MODEL_DIR environment variable not set")
    if not os.getenv("NLTK_DATA"):
        raise SkipTest("NLTK_DATA environment variable not set")

    # initialize the syntax wrapper
    wrapper = SyntaxParserWrapper()

    # read in the input file
    input_file = MY_DIR / "data" / "rst_document.txt"
    with open(input_file, 'r') as docfh:

        # create the document dictionary
        doc_dict = {"raw_text": docfh.read(), "doc_id": "test"}

        # parse the document
        trees, starts_paragraph_list = wrapper.parse_document(doc_dict)

        # check that there are a total of 14 parses and 14 paragraph indicators
        eq_(len(trees), 14)
        eq_(len(starts_paragraph_list), 14)

        # check one of the trees to make sure it matches as expected
        expected_tree_5_str = """(S
                                  (PP (IN In) (NP (CD 2000)))
                                  (, ,)
                                  (NP
                                    (NP (NNP Daniel) (NNP Marcu))
                                    (, ,)
                                    (NP (NP (RB also)) (PP (IN of) (NP (NNP ISI))))
                                    (, ,))
                                  (VP
                                    (VBD demonstrated)
                                    (SBAR
                                      (IN that)
                                      (S
                                        (NP
                                          (JJ practical)
                                          (NN discourse)
                                          (NN parsing)
                                          (CC and)
                                          (NN text)
                                          (NN summarization))
                                        (ADVP (RB also))
                                        (VP
                                          (MD could)
                                          (VP
                                            (VB be)
                                            (VP (VBN achieved) (S (VP (VBG using) (NP (NNP RST))))))))))
                                  (. .))
                            """
        expected_tree_5 = ParentedTree.fromstring(expected_tree_5_str)
        eq_(trees[5], expected_tree_5)

        # check that the the paragraph indicates are correct
        expected_starts_paragraphs_list = [True, True, False, False,
                                           False, True, True, False, False,
                                           False, True, False, False, False]
        eq_(starts_paragraph_list, expected_starts_paragraphs_list)
