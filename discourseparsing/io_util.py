"""
Functions for input/output.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""
import logging

import cchardet


def read_text_file(input_path):
    """
    Read text, using cchardet to identify the encoding.

    Parameters
    ----------
    input_path : str
        Path to the input file to read.

    Returns
    -------
    contents: str
        Contents of the input file.
    """
    # read in the contents of the file as bytes first
    with open(input_path, "rb") as input_file:
        doc = input_file.read()

        # decode as utf-8 first; if that doesn't work, use
        # cchardet to auto-detect the encoding
        try:
            doc = doc.decode('utf-8')
        except UnicodeDecodeError:
            chardet_output = cchardet.detect(doc)
            encoding = chardet_output['encoding']
            encoding_confidence = chardet_output['confidence']
            logging.debug(f"decoding {input_path} as {encoding} with "
                          f"{encoding_confidence} confidence")
            doc = doc.decode(encoding)

    return doc
