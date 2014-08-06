#!/usr/bin/env python3

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import math

from discourseparsing.discourse_parsing import Parser
from discourseparsing.discourse_segmentation import Segmenter
from discourseparsing.parse_util import SyntaxParserWrapper
from discourseparsing.rst_parse import segment_and_parse
from discourseparsing.tree_util import TREE_PRINT_MARGIN


def batch_process(docs, output_path, zpar_model_directory,
                  segmentation_model, parsing_model):
    '''
    docs is a list or tuple of (doc_id, text) tuples.
    '''
    syntax_parser = SyntaxParserWrapper(zpar_model_directory)
    segmenter = Segmenter(segmentation_model)

    parser = Parser(max_acts=1, max_states=1, n_best=1)
    parser.load_model(parsing_model)

    with open(output_path, 'w') as outfile:
        for doc_id, text in docs:
            logging.info('doc_id: {}'.format(doc_id))
            doc_dict = {"doc_id": doc_id, "raw_text": text}
            edu_tokens, complete_trees = \
                segment_and_parse(doc_dict, syntax_parser, segmenter, parser)
            print(json.dumps({"doc_id": doc_id, "edu_tokens": edu_tokens, \
                "scored_rst_trees": \
                [{"score": tree["score"],
                  "tree": tree["tree"].pprint(margin=TREE_PRINT_MARGIN)}
                 for tree in complete_trees]}), file=outfile)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--segmentation_model',
                        help='Path to segmentation model.',
                        required=True)
    parser.add_argument('-p', '--parsing_model',
                        help='Path to RST parsing model.',
                        required=True)
    parser.add_argument('-v', '--verbose',
                        help='Print more status information. For every ' +
                        'additional time this flag is specified, ' +
                        'output gets more verbose.',
                        default=0, action='count')
    parser.add_argument('-m', '--max_workers', type=int, default=cpu_count(),
                        help='number of parallel processes to use')
    parser.add_argument('-zm', '--zpar_model_directory', default=None)
    parser.add_argument('input_file', help='json file with a dictionary from' +
                        ' IDs to texts.')
    parser.add_argument('output_prefix', help='path prefix for where outputs' +
                        ' will be stored.')
    args = parser.parse_args()

    # Convert verbose flag to actually logging level.
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]
    # Make warnings from built-in warnings module get formatted more nicely.
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s - %(name)s - %(levelname)s - ' +
                                '%(message)s'), level=log_level)

    with open(args.input_file) as f:
        docs = list(json.load(f).items())
    chunk_size = math.ceil(len(docs) / args.max_workers)
    docs_batches = [docs[i:i + chunk_size]
                    for i in range(0, len(docs), chunk_size)]

    with ProcessPoolExecutor(max_workers=len(docs_batches)) as executor:
        futures = []
        for i, docs_batch in enumerate(docs_batches):
            logging.info('batch size {}'.format(len(docs_batch)))
            output_path = '{}.{}'.format(args.output_prefix, i)
            # batch_process(docs_batch, output_path, args.zpar_model_directory, args.segmentation_model, args.parsing_model)
            future = executor.submit(batch_process, docs_batch,
                                     output_path,
                                     args.zpar_model_directory,
                                     args.segmentation_model,
                                     args.parsing_model)
            futures.append(future)

        # wait for all the results
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
