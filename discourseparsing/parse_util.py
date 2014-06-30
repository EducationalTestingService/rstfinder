
import subprocess
import shlex
import logging
import re
from tempfile import NamedTemporaryFile

import nltk.data

from discourseparsing.tree_util import ParentedTree, convert_parens_to_ptb_format


class SyntaxParserWrapper():
    def __init__(self, zpar_directory='zpar'):
        self.zpar_directory = zpar_directory
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def parse_document(self, doc):
        logging.info('syntax parsing...')

        # TODO replace this with a server and/or a ctypes wrapper

        # zpar.en expects one sentence per line from stdin
        tmpfile = NamedTemporaryFile('w')
        doc = re.sub(r'\s+', r' ', doc.strip())
        sentences = [convert_parens_to_ptb_format(s)
                     for s in self.tokenizer.tokenize(doc)]
        print('\n'.join(sentences), file=tmpfile)
        tmpfile.flush()

        # import sys
        # print('temp file:{}'.format(tmpfile.name), file=sys.stderr)

        zpar_directory = (self.zpar_directory + "/" if not(self.zpar_directory.endswith("/")) else self.zpar_directory)
        zpar_command = zpar_directory + "dist/zpar.en " + zpar_directory + "english -oc {}".format(tmpfile.name)
        zpar_output = subprocess.check_output(shlex.split(zpar_command)).decode('utf-8')

        tmpfile.close()

        # zpar.en outputs constituent trees, one per line, with the "-oc" option
        # the first 3 and last 2 lines are stuff that should be on stderr
        res = [ParentedTree(s) for s in zpar_output.strip().split('\n')[3:-2]]
        #logging.info('syntax parsing results: {}'.format(res))

        return res
