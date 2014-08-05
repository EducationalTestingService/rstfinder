
import cchardet
import logging


def read_text_file(input_path):
    '''
    A function to read text, using cchardet to identify the encoding.
    '''
    with open(input_path, 'rb') as input_file:
        doc = input_file.read()
        chardet_output = cchardet.detect(doc)
        encoding = chardet_output['encoding']
        encoding_confidence = chardet_output['confidence']
        logging.debug('decoding {} as {} with {} confidence'
                      .format(input_path, encoding, encoding_confidence))
        doc = doc.decode(encoding)
    return doc
