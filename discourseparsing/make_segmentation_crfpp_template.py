#!/usr/bin/env python3

'''
This generates a feature template file for CRF++.
See http://crfpp.googlecode.com/svn/trunk/doc/index.html.
It will need to be rerun if new features are added to the segmenter.
'''

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path',
        help='A path to where the CRF++ template file should be created.',
        default='segmentation_crfpp_template.txt')
    parser.add_argument('--num_features', type=int, default=13)
    args = parser.parse_args()

    with open(args.output_path, 'w') as outfile:
        for i in range(args.num_features):
            for j in [-2, -1, 0, 1, 2]:
                print('U{:03d}{}:%x[{},{}]'.format(i, j + 2, j, i),
                      file=outfile)
            print(file=outfile)


if __name__ == '__main__':
    main()
