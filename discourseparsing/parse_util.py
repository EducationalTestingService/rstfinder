"""
Utility classes and functions for syntactic parsing.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""
import logging
import os
import re
import socket
import xmlrpc.client

import nltk.data
from nltk.tree import ParentedTree
from zpar import ZPar

from .paragraph_splitting import ParagraphSplitter
from .tree_util import TREE_PRINT_MARGIN, convert_parens_to_ptb_format


class SyntaxParserWrapper():
    """A wrapper class around the syntactic parser."""

    def __init__(self, zpar_model_directory=None, hostname=None, port=None):
        """
        Initialize the parser wrapper.

        Parameters
        ----------
        zpar_model_directory : str, optional
            The path to the directory containing the ZPar constituency model.
        hostname : str, optional
            The name of the machine on which the ZPar server is running, if any.
        port : int, optional
            The port at which the ZPar server is running, if any.

        Raises
        ------
        OSError
            If ZPar couldn't be loaded.
        """
        self.zpar_model_directory = zpar_model_directory
        if self.zpar_model_directory is None:
            self.zpar_model_directory = os.getenv("ZPAR_MODEL_DIR",
                                                  "zpar/english")

        # TODO: allow pre-tokenized input
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._zpar_proxy = None
        self._zpar_ref = None

        # if a port is specified, then we want to use the server
        if port:

            # if no hostname was specified, then try the local machine
            hostname = "localhost" if hostname is None else hostname
            logging.info(f"Connecting to zpar server at {hostname}:{port} ...")

            # see if a server actually exists
            connected, server_proxy = self._get_rpc(hostname, port)
            if connected:
                self._zpar_proxy = server_proxy
            else:
                logging.warning('Could not connect to zpar server.')

        # otherwise, we want to use the python zpar module
        else:

            logging.info("Trying to locate zpar shared library ...")
            try:
                # Create a zpar wrapper data structure
                z = ZPar(self.zpar_model_directory)
                self._zpar_ref = z.get_parser()
            except OSError as e:
                logging.warning("Could not load zpar via python-zpar."
                                "Did you set `ZPAR_MODEL_DIR` correctly?")
                raise e

    @staticmethod
    def _get_rpc(hostname, port):
        """Get the zpar server proxy, if one exists."""
        proxy = xmlrpc.client.ServerProxy(f"http://{hostname}:{port}",
                                          use_builtin_types=True,
                                          allow_none=True)

        # call an empty method just to check that the server exists.
        try:
            proxy._()
        except xmlrpc.client.Fault:
            # this call is expected to raise a Fault, so just pass
            pass
        except socket.error:
            # if no server was found, indicate so...
            return False, None

        # Otherwise, return that a server was found, and return its proxy.
        return True, proxy

    def tokenize_document(self, text):
        """Tokenize given document into sentences."""
        tmpdoc = re.sub(r"\s+", r' ', text.strip())
        sentences = [convert_parens_to_ptb_format(s)
                     for s in self.tokenizer.tokenize(tmpdoc)]
        return sentences

    def _parse_document_via_server(self, text, doc_id):
        """Parse given text using the ZPar server."""
        # sentence tokenize the text
        sentences = self.tokenize_document(text)
        res = []

        # iterate over each sentence and parse
        for sentence in sentences:
            parsed_sent = self._zpar_proxy.parse_sentence(sentence)
            if parsed_sent:
                res.append(ParentedTree.fromstring(parsed_sent))
            else:
                logging.warning(f"The syntactic parser was unable to parse: "
                                f"{sentence}, doc_id = {doc_id}")
        tree_list = [tree.pformat(margin=TREE_PRINT_MARGIN) for tree in res]
        logging.debug(f"syntax parsing results: {tree_list}")
        return res

    def _parse_document_via_lib(self, text, doc_id):
        """Parse given text using the compiled ZPar library."""
        # sentence tokenize the text
        sentences = self.tokenize_document(text)
        res = []

        # iterate over each sentence and parse
        for sentence in sentences:
            parsed_sent = self._zpar_ref.parse_sentence(sentence)
            if parsed_sent:
                res.append(ParentedTree.fromstring(parsed_sent))
            else:
                logging.warning(f"The syntactic parser was unable to parse: "
                                f"{sentence}, doc_id = {doc_id}")
        tree_list = [tree.pformat(margin=TREE_PRINT_MARGIN) for tree in res]
        logging.debug(f"syntax parsing results: {tree_list}")
        return res

    def parse_document(self, doc_dict):
        """
        Produce a constituency parse for the given document.

        Parameters
        ----------
        doc_dict : dict
            A dictionary representing the document to parse.

        Returns
        -------
        results : tuple
            A tuple containing the constituency trees for each sentence
            in the document and a list of booleans indicating which of the
            trees are for sentences that appear at the begininng of a
            paragraph.

        Raises
        ------
        RuntimeError
            If the ZPar server is to be used for parsing but is not
            available.
        """
        doc_id = doc_dict["doc_id"]
        logging.info(f"syntax parsing, doc_id = {doc_id}")

        # get the paragraphs in this document
        # TODO: should there be some extra preprocessing to deal with fancy
        # quotes, etc.? The tokenizer doesn't appear to handle it well
        paragraphs = ParagraphSplitter.find_paragraphs(doc_dict["raw_text"],
                                                       doc_id=doc_id)

        starts_paragraph_list = []
        trees = []
        no_parse_for_paragraph = False

        # iterate over each found paragraph
        for paragraph in paragraphs:

            # try to use the server first
            if self._zpar_proxy:
                parse_trees = self._parse_document_via_server(paragraph, doc_id)

            # then fall back to the shared library
            else:
                if self._zpar_ref is None:
                    raise RuntimeError('The ZPar server is unavailable.')
                parse_trees = self._parse_document_via_lib(paragraph, doc_id)

            if len(parse_trees) > 0:
                starts_paragraph_list.append(True)
                starts_paragraph_list.extend([False for t in parse_trees[1:]])
                trees.extend(parse_trees)
            else:
                # TODO: add some sort of error flag to the dictionary
                # for this document?
                no_parse_for_paragraph = True

        logging.debug(f"starts_paragraph_list = {starts_paragraph_list}, "
                      f"doc_id = {doc_id}")

        # make sure that either the number of `True` indicators in
        # `starts_paragraph_list` equals the number of paragraphs, or
        # that the parser had to skip a paragraph entirely
        assert (sum(starts_paragraph_list) == len(paragraphs) or
                no_parse_for_paragraph)
        assert len(trees) == len(starts_paragraph_list)

        return trees, starts_paragraph_list
