#!/usr/bin/env python3

import argparse
import logging
import sys

sys.path.append('/Users/nmadnani/work/discourse-parsing')

from discourseparsing.parse_util import SyntaxParserWrapper



if __name__ == '__main__':

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='test_parse_util.py')
    parser.add_argument('--input', dest='inputfile', help="Input file", required=True)
    parser.add_argument('--zpar', dest='zpar_directory', help="ZPar directory", required=True)
    parser.add_argument('--port', dest='port', type=int,
                        help="hostname for already running zpar server",
                        default=None,
                        required=False)
    parser.add_argument('--host', dest='hostname',
                        help="port for already running zper server",
                        default=None,
                        required=False)

    # parse given command line arguments
    args = parser.parse_args()

    # set up the logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # initialize the syntax wrapper
    wrapper = SyntaxParserWrapper(zpar_directory=args.zpar_directory,
                                  hostname=args.hostname,
                                  port=args.port)
    with open(args.inputfile, 'r') as docf:
        output = wrapper.parse_document(docf.read())
        print(output)
