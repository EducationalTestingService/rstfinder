
import subprocess
import shlex
import re
from tempfile import NamedTemporaryFile

import nltk.data

from discourseparsing.tree_util import ParentedTree

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def parse_document(doc):
    # TODO replace this with a server and/or a ctypes wrapper

    # zpar.en expects one sentence per line from stdin
    tmpfile = NamedTemporaryFile('w')
    doc = re.sub(r'\s+', r' ', doc.strip())
    print('\n'.join(tokenizer.tokenize(doc)), file=tmpfile)
    tmpfile.flush()

    # import sys
    # print('temp file:{}'.format(tmpfile.name), file=sys.stderr)

    zpar_output = subprocess.check_output(
        shlex.split('zpar/dist/zpar.en zpar/english -oc {}'.format(tmpfile.name))).decode('utf-8')

    tmpfile.close()

    # zpar.en outputs constituent trees, one per line, with the "-oc" option
    # the first 3 and last 2 lines are stuff that should be on stderr
    return [ParentedTree(s) for s in zpar_output.strip().split('\n')[3:-2]]
