
import ctypes as c
import socket
import logging
import re
import os
import xmlrpc.client

import nltk.data
from nltk.tree import ParentedTree

from discourseparsing.tree_util import (convert_parens_to_ptb_format,
                                        TREE_PRINT_MARGIN)
from discourseparsing.paragraph_splitting import ParagraphSplitter


class SyntaxParserWrapper():
    def __init__(self, zpar_model_directory='zpar/english', hostname=None,
                 port=None):
        self.zpar_model_directory = zpar_model_directory
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._zpar_proxy = None
        self._zpar_ref = None

        # if a port is specified, then we want to use the server
        if port:

            # if no hostname was specified, then try the local machine
            hostname = 'localhost' if not hostname else hostname
            logging.info('Trying to connect to zpar server at {}:{} ...'
                         .format(hostname, port))

            # try to see if a server actually exists
            connected, server_proxy = self._get_rpc(hostname, port)
            if connected:
                self._zpar_proxy = server_proxy
            else:
                logging.warning('Could not connect to zpar server')

        # otherwise, we want to use the python zpar module
        else:

            logging.info('Trying to locate zpar shared library ...')

            # get the path to the zpar shared library via the environment
            # variable
            zpar_library_dir = os.getenv('ZPAR_LIBRARY_DIR', '')
            zpar_library_path = os.path.join(zpar_library_dir, 'zpar.so')

            try:
                # Create a zpar wrapper data structure
                self._zpar_ref = c.cdll.LoadLibrary(zpar_library_path)
            except OSError as e:
                logging.warning('Could not find zpar shared library. ' +
                                'Did you set ZPAR_LIBRARY_DIR correctly?')
                raise e
            else:
                self._initialize_zpar()

    def __del__(self):
        if self._zpar_ref:
            unload_models = self._zpar_ref.unload_models
            unload_models.restype = None
            unload_models()

    @staticmethod
    def _get_rpc(hostname, port):
        '''
        Tries to get the zpar server proxy, if one exists.
        '''

        proxy = xmlrpc.client.ServerProxy(
            'http://{}:{}'.format(hostname, port),
            use_builtin_types=True, allow_none=True)
        # Call an empty method just to check that the server exists.
        try:
            proxy._()
        except xmlrpc.client.Fault:
            # The above call is expected to raise a Fault, so just pass here.
            pass
        except socket.error:
            # If no server was found, indicate so...
            return False, None

        # Otherwise, return that a server was found, and return its proxy.
        return True, proxy

    def _initialize_zpar(self):
        # define the argument and return types for all
        # the functions we want to expose to the client
        load_parser = self._zpar_ref.load_parser
        load_parser.restype = c.c_int
        load_parser.argtypes = [c.c_char_p]

        parse_sentence = self._zpar_ref.parse_sentence
        parse_sentence.restype = c.c_char_p
        parse_sentence.argtypes = [c.c_char_p]

        if load_parser(self.zpar_model_directory.encode('utf-8')):
            self._zpar_ref.unload_models()
            raise Exception('Cannot find parser model at {}'
                            .format(self.zpar_model_directory))

    def tokenize_document(self, doc):
        tmpdoc = re.sub(r'\s+', r' ', doc.strip())
        sentences = [convert_parens_to_ptb_format(s)
                     for s in self.tokenizer.tokenize(tmpdoc)]
        return sentences

    def _parse_document_via_server(self, doc):
        sentences = self.tokenize_document(doc)
        res = []
        for sentence in sentences:
            parsed_sent = self._zpar_proxy.parse_sentence(sentence)
            if parsed_sent:
                res.append(ParentedTree.fromstring(parsed_sent))
            else:
                logging.warning('The syntactic parser was unable to parse: {}'
                                .format(sentence))
        logging.debug('syntax parsing results: {}'.format(
            [t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res

    def _parse_document_via_lib(self, doc):
        sentences = self.tokenize_document(doc)
        res = []
        for sentence in sentences:
            parsed_sent = self._zpar_ref.parse_sentence(
                sentence.encode("utf-8"))
            if parsed_sent:
                res.append(ParentedTree.fromstring(parsed_sent.decode('utf-8')))
            else:
                logging.warning('The syntactic parser was unable to parse: {}'
                                .format(sentence))
        logging.debug('syntax parsing results: {}'.format(
            [t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res

    def parse_document(self, doc):
        logging.info('syntax parsing...')

        # TODO should there be some extra preprocessing to deal with fancy quotes, etc.?
        # The tokenizer doesn't appear to handle it well
        paragraphs = ParagraphSplitter.find_paragraphs(doc)

        starts_paragraph_list = []
        trees = []
        no_parse_for_paragraph = False
        for paragraph in paragraphs:
            # try to use the server first
            if self._zpar_proxy:
                trees_p = self._parse_document_via_server(paragraph)
            # then fall back to the shared library
            else:
                if self._zpar_ref is None:
                    raise RuntimeError('The ZPar server is unavailable.')
                trees_p = self._parse_document_via_lib(paragraph)

            if len(trees_p) > 0:
                starts_paragraph_list.append(True)
                starts_paragraph_list.extend([False for t in trees_p[1:]])
                trees.extend(trees_p)
            else:
                # TODO add some sort of error flag to the dictionary for this document?
                no_parse_for_paragraph = True

        logging.debug('starts_paragraph_list = {}'
                      .format(starts_paragraph_list))

        # Check that either the number of True indicators in
        # starts_paragraph_list equals the number of paragraphs, or that the
        # syntax parser had to skip a paragraph entirely.
        assert (sum(starts_paragraph_list) == len(paragraphs)
                or no_parse_for_paragraph)
        assert len(trees) == len(starts_paragraph_list)

        return trees, starts_paragraph_list


