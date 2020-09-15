#!/usr/bin/env python3

import argparse
import logging

from rstfinder.parse_util import SyntaxParserWrapper

if __name__ == '__main__':

    # set up an argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='inputfile', help="Input file",
                        required=True)
    parser.add_argument('--models', dest='zpar_model_directory',
                        help="ZPar model directory", required=True)
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
    wrapper = \
        SyntaxParserWrapper(zpar_model_directory=args.zpar_model_directory,
                            hostname=args.hostname, port=args.port)
    with open(args.inputfile, 'r') as docf:
        doc_dict = {"raw_text": docf.read(), "doc_id": args.input}
        trees, starts_paragraph_list = wrapper.parse_document(doc_dict)
        print("Syntax trees: {}".format(trees))
        print("Tree starts paragraph indicators".format(starts_paragraph_list))
