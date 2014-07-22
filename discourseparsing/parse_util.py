
import ctypes as c
import subprocess
import shlex
import socket
import logging
import re
import os
from tempfile import NamedTemporaryFile
import sys
import xmlrpc.client

import nltk.data
from nltk.tree import ParentedTree

from discourseparsing.tree_util import (convert_parens_to_ptb_format,
                                       TREE_PRINT_MARGIN)

class SyntaxParserWrapper():
    def __init__(self, zpar_directory='zpar', zpar_model_directory=None,
                 hostname=None, port=None):
        self.zpar_directory = zpar_directory
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._zpar_proxy = None
        self._zpar_ref = None

        # if a port is specified, then we want to use the server
        if port:

            # if no hostname was specified, then try the local machine
            hostname = 'localhost' if not hostname else hostname
            logging.info('Trying to connect to zpar server at {}:{} ...'.format(hostname, port))

            # try to see if a server actually exists
            connected, server_proxy = self._get_rpc(hostname, port)
            if connected:
                self._zpar_proxy = server_proxy
            else:
                logging.warning('Could not connect to zpar server')

        # otherwise, we want to use the python zpar module
        else:

            logging.info('Trying to locate zpar shared library ...')

            # get the path to the zpar shared library via the environment variable
            zpar_library_dir = os.getenv('ZPAR_LIBRARY_DIR', '')
            zpar_library_path = os.path.join(zpar_library_dir, 'zpar.so')

            try:
                # Create a zpar wrapper data structure
                self._zpar_ref = c.cdll.LoadLibrary(zpar_library_path)
            except OSError:
                logging.warning('Could not find zpar shared library. Did you set ZPAR_LIBRARY_DIR correctly?')
                logging.warning('Falling back to subprocess mode.')
            else:
                self._initialize_zpar()

    # try to get the zpar server proxy, if one exists
    def _get_rpc(self, hostname, port):

        proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(hostname, port),
                                                         use_builtin_types=True,
                                                         allow_none=True)
        try:
            proxy._()
        except xmlrpc.client.Fault:
            pass
        except socket.error:
            return False, None

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

        zpar_model_directory = os.path.join(self.zpar_directory, 'english')
        if load_parser(zpar_model_directory.encode('utf-8')):
            sys.stderr.write('Cannot find parser model at {}\n'.format(zpar_model_directory))
            self._zpar_ref.unload_models()
            sys.exit(1)

    def tokenize_document(self, doc):
        tmpdoc = re.sub(r'\s+', r' ', doc.strip())
        sentences = [convert_parens_to_ptb_format(s)
                     for s in self.tokenizer.tokenize(tmpdoc)]
        return sentences

    def _parse_document_via_subprocess(self, doc):
        # zpar.en expects one sentence per line from stdin
        tmpfile = NamedTemporaryFile('w')
        sentences = self.tokenize_document(doc)
        print('\n'.join(sentences), file=tmpfile)
        tmpfile.flush()

        logging.debug('parse_util temp file: {}'.format(tmpfile.name))

        zpar_command = '{} {} -oc {}'.format(
            os.path.join(self.zpar_directory, 'dist', 'zpar.en'),
            os.path.join(self.zpar_directory, "english"),
            tmpfile.name)
        zpar_output = subprocess.check_output(
            shlex.split(zpar_command)).decode('utf-8')

        tmpfile.close()

        # zpar.en outputs constituent trees, 1 per line, with the "-oc" option
        # the first 3 and last 2 lines are stuff that should be on stderr
        res = [ParentedTree(s) for s
               in zpar_output.strip().split('\n')[3:-1]]
        logging.debug('syntax parsing results: {}'.format(
            [t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res

    def _parse_document_via_server(self, doc):
        sentences = self.tokenize_document(doc)
        res = []
        for sentence in sentences:
            try:
                parsed_sent = self._zpar_proxy.parse_sentence(sentence)
            except xmlrpc.client.Fault as flt:
                sys.stderr.write("Fault {}: {}\n".format(flt.faultCode,
                                                         flt.faultString))
                sys.exit(1)
            else:
                res.append(ParentedTree(parsed_sent))
                logging.debug('syntax parsing results: {}'.format([t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res

    def _parse_document_via_lib(self, doc):
        sentences = self.tokenize_document(doc)
        res = []
        for sentence in sentences:
            parsed_sent = self._zpar_ref.parse_sentence(sentence.encode("utf-8"))
            res.append(ParentedTree(parsed_sent.decode('utf-8')))
        logging.debug('syntax parsing results: {}'.format([t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res

    def parse_document(self, doc):
        logging.info('syntax parsing...')

        # TODO should there be some extra preprocessing to deal with fancy quotes, etc.?
        # The tokenizer doesn't appear to handle it well

        # try to use the server first
        if self._zpar_proxy:
            return self._parse_document_via_server(doc)
        # then fall back to the shared library
        elif self._zpar_ref:
            return self._parse_document_via_lib(doc)
        # and finally to using subprocess
        else:
            return self._parse_document_via_subprocess(doc)
