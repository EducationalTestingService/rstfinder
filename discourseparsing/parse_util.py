
import subprocess
import shlex
import logging
import re
import os
from tempfile import NamedTemporaryFile

import nltk.data

from discourseparsing.tree_util import (ParentedTree,
                                        convert_parens_to_ptb_format,
                                        TREE_PRINT_MARGIN)


class SyntaxParserWrapper():
    def __init__(self, zpar_directory='zpar'):
        self.zpar_directory = zpar_directory
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def parse_document(self, doc):
        logging.info('syntax parsing...')

        # TODO replace this with a server and/or a ctypes wrapper

        # TODO should there be some extra preprocessing to deal with fancy quotes, etc.?  The tokenizer doesn't appear to handle it well

        # zpar.en expects one sentence per line from stdin
        tmpfile = NamedTemporaryFile('w')
        doc = re.sub(r'\s+', r' ', doc.strip())
        sentences = [convert_parens_to_ptb_format(s)
                     for s in self.tokenizer.tokenize(doc)]
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
               in zpar_output.strip().split('\n')[3:-2]]
        logging.debug('syntax parsing results: {}'.format(
            [t.pprint(margin=TREE_PRINT_MARGIN) for t in res]))

        return res
