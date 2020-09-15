#!/usr/bin/env python
# License: MIT

from setuptools import find_packages, setup


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
      description=('A discourse parser and segmenter for use with the '
                   'Rhetorical Structure Theory Discourse Treebank '
                   '(https://catalog.ldc.upenn.edu/LDC2002T07).'),
      long_description=readme(),
      keywords='discourse parsing rst',
      url='http://github.com/EducationalTestingService/discourse-parsing',
      author='Michael Heilman',
      maintainer='Nitin Madnani',
      maintainer_email='nmadnani@ets.org',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      entry_points={'console_scripts': ['segment_document = discourseparsing.segment_document:main',
                                        'tune_segmentation_model = discourseparsing.tune_segmentation_model:main',
                                        'rst_parse = discourseparsing.rst_parse:main',
                                        'tune_rst_parser = discourseparsing.tune_rst_parser:main',
                                        'convert_rst_discourse_tb = discourseparsing.convert_rst_discourse_tb:main',
                                        'make_traindev_split = discourseparsing.make_traindev_split:main',
                                        'rst_eval = discourseparsing.rst_eval:main',
                                        'extract_segmentation_features = discourseparsing.extract_segmentation_features:main',
                                        'rst_parse_batch = discourseparsing.rst_parse_batch:main',
                                        'compute_bootstrap_from_predictions = discourseparsing.utils.compute_bootstrap_from_predictions:main',
                                        'visualize_rst_tree = discourseparsing.utils.visualize_rst_tree:main']},
      install_requires=requirements(),
      zip_safe=False)
