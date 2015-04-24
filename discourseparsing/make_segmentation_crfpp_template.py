#!/usr/bin/env python
# License: MIT

'''
This generates a feature template file for CRF++.
See http://crfpp.googlecode.com/svn/trunk/doc/index.html.
It needs to be rerun when the segmenter feature set changes.
'''

import argparse


def make_segmentation_crfpp_template(output_path, num_features=12):
    with open(output_path, 'w') as outfile:
        for i in range(num_features):
            # This makes it so the features for the current word are based
            # on the current word and the previous 2 and next 2 words.
            for j in [-2, -1, 0, 1, 2]:
                print('U{:03d}{}:%x[{},{}]'.format(i, j + 2, j, i),
                      file=outfile)
            print(file=outfile)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output_path',
        help='A path to where the CRF++ template file should be created.',
        default='segmentation_crfpp_template.txt')
    parser.add_argument('--num_features', type=int, default=12)
    args = parser.parse_args()
    make_segmentation_crfpp_template(args.output_path, args.num_features)


if __name__ == '__main__':
    main()
