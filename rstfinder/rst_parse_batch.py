#!/usr/bin/env python
"""
Functions for batch-parsing multiple documents in parallel.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""
import argparse
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from .discourse_parsing import Parser
from .discourse_segmentation import Segmenter
from .parse_util import SyntaxParserWrapper
from .rst_parse import segment_and_parse
from .tree_util import TREE_PRINT_MARGIN


def batch_process(docs,
                  output_path,
                  zpar_model_directory,
                  segmentation_model,
                  parsing_model):
    """
    Produce RST parse trees for a given set of documents.

    The trees along with scores are printed to STDOUT.

    Parameters
    ----------
    docs : list
        List of (doc_id, text) tuples.
    output_path : str
        Path where the output will be stored.
    zpar_model_directory : str
        Path to the directory containing the ZPar English constituency model.
    segmentation_model : str
        Path to the CRF++ based discourse segmentation model.
    parsing_model : str
        Path to the RST parsing model.
    """
    # initialize the wrapper for the constituency parser
    syntax_parser = SyntaxParserWrapper(zpar_model_directory)

    # initialize the discourse segmenter
    segmenter = Segmenter(segmentation_model)

    # initialize the RST parser container and load the model
    parser = Parser(max_acts=1, max_states=1, n_best=1)
    parser.load_model(parsing_model)

    # process each document and write out its output
    with open(output_path, 'w') as outfile:
        for doc_id, text in docs:
            logging.info(f"doc_id: {doc_id}")
            doc_dict = {"doc_id": doc_id, "raw_text": text}
            (edu_tokens,
             complete_trees) = segment_and_parse(doc_dict,
                                                 syntax_parser,
                                                 segmenter,
                                                 parser)
            print(json.dumps({"doc_id": doc_id,
                              "edu_tokens": edu_tokens,
                              "scored_rst_trees": [{"score": tree["score"],
                                                    "tree": tree["tree"].pformat(margin=TREE_PRINT_MARGIN)}
                                                   for tree in complete_trees]}),
                  file=outfile)


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g",
                        "--segmentation_model",
                        help="Path to segmentation model.",
                        required=True)
    parser.add_argument("-p",
                        "--parsing_model",
                        help="Path to RST parsing model.",
                        required=True)
    parser.add_argument("-v",
                        "--verbose",
                        help="Print more status information. For every "
                             "additional time this flag is specified, "
                             "output gets more verbose.",
                        default=0,
                        action="count")
    parser.add_argument("-m",
                        "--max_workers",
                        type=int,
                        default=cpu_count(),
                        help="Number of parallel processes to use")
    parser.add_argument("-zm",
                        "--zpar_model_directory",
                        help="Path to the directory containing the "
                             "ZPar English constituency model.",
                        default=None)
    parser.add_argument("input_file",
                        help="JSON file with a dictionary from IDs to texts.")
    parser.add_argument("output_prefix",
                        help="Path prefix for where outputs will be stored.")
    args = parser.parse_args()

    # convert verbose flag to logging level
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, 2)]

    # format warnings more nicely
    logging.captureWarnings(True)
    logging.basicConfig(format=("%(asctime)s - %(name)s - %(levelname)s - "
                                "%(message)s"),
                        level=log_level)

    # create batches of documents
    with open(args.input_file) as inputfh:
        docs = list(json.load(inputfh).items())
    chunk_size = math.ceil(len(docs) / args.max_workers)
    docs_batches = [docs[i:i + chunk_size]
                    for i in range(0, len(docs), chunk_size)]

    # process each batch of documents in parallel
    with ProcessPoolExecutor(max_workers=len(docs_batches)) as executor:
        futures = []
        for i, docs_batch in enumerate(docs_batches):
            logging.info(f"batch size: {len(docs_batch)}")
            output_path = f"{args.output_prefix}.{i}"
            future = executor.submit(batch_process,
                                     docs_batch,
                                     output_path,
                                     args.zpar_model_directory,
                                     args.segmentation_model,
                                     args.parsing_model)
            futures.append(future)

        # wait for all the results
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
