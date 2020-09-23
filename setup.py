#!/usr/bin/env python

from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    pass


setup(name='rstfinder',
      version='0.2.1',
      description=('A discourse parser and segmenter for use with the '
                   'Rhetorical Structure Theory Discourse Treebank '
                   '(https://catalog.ldc.upenn.edu/LDC2002T07).'),
      long_description=readme(),
      keywords='discourse parsing rst',
      url='http://github.com/EducationalTestingService/rstfinder',
      author='Michael Heilman',
      maintainer='Nitin Madnani',
      maintainer_email='nmadnani@ets.org',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      entry_points={'console_scripts': ['segment_document = rstfinder.segment_document:main',
                                        'tune_segmentation_model = rstfinder.tune_segmentation_model:main',
                                        'rst_parse = rstfinder.rst_parse:main',
                                        'tune_rst_parser = rstfinder.tune_rst_parser:main',
                                        'convert_rst_discourse_tb = rstfinder.convert_rst_discourse_tb:main',
                                        'make_traindev_split = rstfinder.make_traindev_split:main',
                                        'rst_eval = rstfinder.rst_eval:main',
                                        'extract_segmentation_features = rstfinder.extract_segmentation_features:main',
                                        'rst_parse_batch = rstfinder.rst_parse_batch:main',
                                        'compute_bootstrap_from_predictions = rstfinder.utils.compute_bootstrap_from_predictions:main',
                                        'try_head_rules = rstfinder.utils.try_head_rules:main',
                                        'visualize_rst_tree = rstfinder.utils.visualize_rst_tree:main']},
      install_requires=requirements(),
      zip_safe=False)
