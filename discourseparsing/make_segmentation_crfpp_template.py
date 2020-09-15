#!/usr/bin/env python
"""
Generate a feature template for CRF++ segmentation model.

This script generates a feature template file for CRF++.
See http://taku910.github.io/crfpp/.

This script will needs to be re-run when the segmenter
feature set changes.

:author: Michael Heilman
:author: Nitin Madnani
:organization: ETS
"""

import argparse


def make_segmentation_crfpp_template(output_path, num_features=12):
    """
    Create feature template and write to ``output_path``.

    Parameters
    ----------
    output_path : str
        Path to output file where feature template is written.
    num_features : int, optional
        Number of features to extract.
        Defaults to 12.
    """
    with open(output_path, 'w') as outfile:
        for i in range(num_features):
            # this template says that the features for the current word are
            # based on the current word and the previous 2 and next 2 words
            for j in [-2, -1, 0, 1, 2]:
                print(f"U{i:03d}{j+2}:%x[{j},{i}]", file=outfile)
            print(file=outfile)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_path",
                        help="Path to the output CRF++ template file.",
                        default="segmentation_crfpp_template.txt")
    parser.add_argument("--num_features",
                        type=int,
                        default=12)
    args = parser.parse_args()
    make_segmentation_crfpp_template(args.output_path, args.num_features)


if __name__ == "__main__":
    main()
