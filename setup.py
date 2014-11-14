#!/usr/bin/env python
# License: MIT

from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    req_path = 'requirements.txt'
    with open(req_path) as f:
        reqs = f.read().splitlines()
    return reqs


setup(name='discourseparsing',
      version='0.2.1',
      description=('A discourse parser and segmenter for use with the \
                    Rhetorical Structure Theory Discourse Treebank \
                    (https://catalog.ldc.upenn.edu/LDC2002T07).'),
      long_description=readme(),
      keywords='discourse parsing rst',
      url='http://github.com/mheilman/discourse-parsing',
      author='Michael Heilman',
      author_email='mheilman@ets.org',
      license='MIT',
      packages=['discourseparsing'],
      entry_points={'console_scripts': ['segment_document = discourseparsing.segment_document:main',
                                        'tune_segmentation_model = discourseparsing.tune_segmentation_model:main',
                                        'rst_parse = discourseparsing.rst_parse:main',
                                        'tune_rst_parser = discourseparsing.tune_rst_parser:main',
                                        'convert_rst_discourse_tb = discourseparsing.convert_rst_discourse_tb:main',
                                        'rst_eval = discourseparsing.rst_eval:main',
                                        'extract_segmentation_features = discourseparsing.extract_segmentation_features:main',
                                        'rst_parse_batch = discourseparsing.rst_parse_batch:main']},
      install_requires=requirements())
