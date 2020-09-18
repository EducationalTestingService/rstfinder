import json
import os
import unittest
from pathlib import Path

from nltk.tree import ParentedTree
from nose import SkipTest
from nose.tools import eq_, ok_
from rstfinder.discourse_parsing import Parser
from rstfinder.discourse_segmentation import Segmenter
from rstfinder.parse_util import SyntaxParserWrapper
from rstfinder.rst_parse import segment_and_parse


class TestSegmentAndParse(unittest.TestCase):
    """Test the `segment_and_parse()` function from `rst_parse.py`."""

    @classmethod
    def setUpClass(cls):  # noqa: D102
        cls.currdir = Path(__file__).parent
        cls.parser = None
        cls.segmenter = None
        cls.rst_parser = None
        cls.document_file = cls.currdir / "data" / "rst_document.txt"
        cls.partial_doc_dict_file = cls.currdir / "data" / "partial_doc_dict.json"

        segmenter_model_path = cls.currdir / "models" / "segmenter.model"
        if segmenter_model_path.exists():
            cls.segmenter = Segmenter(str(segmenter_model_path))

        if os.getenv("ZPAR_MODEL_DIR") and os.getenv("NLTK_DATA"):
            cls.parser = SyntaxParserWrapper()

        rst_model_path = (cls.currdir / "models" /
                          "rst_parsing_all_feats_LogisticRegression.model")
        if rst_model_path.exists():
            cls.rst_parser = Parser(1, 1, 1)
            cls.rst_parser.load_model(str(cls.currdir / "models"))

    def test_segment_and_parse(self):
        """Test segmentation + rst parsing pipeline."""
        if (self.parser is None or
                self.segmenter is None or
                self.rst_parser is None):
            raise SkipTest("one or more models could not be found")

        # create the document dictionary and run the segmenter+parser
        with open(self.document_file, 'r') as docfh:
            doc_dict = {"raw_text": docfh.read(), "doc_id": "test"}
            edu_tokens, rst_trees = segment_and_parse(doc_dict,
                                                      self.parser,
                                                      self.segmenter,
                                                      self.rst_parser)

        # check that the document dictionary now has other fields populated
        ok_("tokens" in doc_dict)
        eq_(len(doc_dict["tokens"]), 14)
        ok_("token_tree_positions" in doc_dict)
        eq_(len(doc_dict["token_tree_positions"]), 14)
        ok_("pos_tags" in doc_dict)
        eq_(len(doc_dict["pos_tags"]), 14)
        ok_("syntax_trees" in doc_dict)
        eq_(len(doc_dict["syntax_trees"]), 14)
        ok_("edu_start_indices" in doc_dict)
        eq_(len(doc_dict["edu_start_indices"]), 32)
        ok_("edu_starts_paragraph" in doc_dict)
        eq_(len(doc_dict["edu_starts_paragraph"]), 32)

        # make sure there are the expected number of EDUs
        # and check the first and last EDUs as well
        eq_(len(edu_tokens), 32)
        eq_(edu_tokens[0], ['Rhetorical', 'structure', 'theory'])
        eq_(edu_tokens[-1], ['where', 'a', 'satellites', 'have', 'been',
                             'deleted', 'can', 'be', 'understood', 'to',
                             'a', 'certain', 'extent', '.'])

        # check that we got one tree back
        rst_tree_list = list(rst_trees)
        eq_(len(rst_tree_list), 1)
        tree = rst_tree_list[0]['tree']
        isinstance(tree, ParentedTree)
        eq_(tree[0].label(), 'nucleus:span')

    def test_segment_and_parse_partial(self):
        """Test pipeline with partially populated document dictionary."""
        if self.segmenter is None or self.rst_parser is None:
            raise SkipTest("one or more models could not be found")

        with open(self.partial_doc_dict_file, "r") as docdictfh:
            doc_dict = json.load(docdictfh)

        edu_tokens, rst_trees = segment_and_parse(doc_dict,
                                                  self.parser,
                                                  self.segmenter,
                                                  self.rst_parser)

        # check that the document dictionary now has other fields populated
        ok_("edu_start_indices" in doc_dict)
        eq_(len(doc_dict["edu_start_indices"]), 5)
        eq_(doc_dict["edu_start_indices"], [(0, 0, 0),
                                            (0, 3, 1),
                                            (0, 6, 2),
                                            (0, 25, 3),
                                            (0, 28, 4)])
        ok_("edu_starts_paragraph" in doc_dict)
        eq_(len(doc_dict["edu_starts_paragraph"]), 5)
        eq_(doc_dict["edu_starts_paragraph"], [True, False, False, False, False])

        # make sure there are the expected number of EDUs
        # and check the first and last EDUs as well
        eq_(len(edu_tokens), 5)
        eq_(edu_tokens[0], ['Rhetorical', 'structure', 'theory'])
        eq_(edu_tokens[-1], ['and', 'defined', 'in', 'a', 'seminal',
                             'paper', 'in', '1988', '.'])

    def test_segment_and_parse_no_text(self):
        """Test that parse pipeline fails gracefully with no text."""
        doc_dict = {"raw_text": "   ", "doc_id": "blank"}
        edu_tokens, rst_trees = segment_and_parse(doc_dict,
                                                  self.parser,
                                                  self.segmenter,
                                                  self.rst_parser)
        eq_(edu_tokens, [])
        eq_(rst_trees, [])
