#!/usr/bin/env python3

from discourseparsing.tune_segmentation_model import \
    convert_crfpp_output, evaluate_segmentation_output


def test_segmentation_evaluation():
    '''
    Checks that the segmentation evaluation works as expected.
    '''
    output = ("foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n" "foo\tC-EDU\tC-EDU\n"
              "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tB-EDU\n" "foo\tC-EDU\tB-EDU\n"
              "foo\tC-EDU\tC-EDU\n" "foo\tC-EDU\tC-EDU\n" "\n"
              "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n" "foo\tB-EDU\tC-EDU\n"
              "foo\tC-EDU\tC-EDU\n" "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n")
    converted = convert_crfpp_output(output)
    precision, recall, f1, num_gold, num_pred = \
        evaluate_segmentation_output(converted)

    assert precision == 2.0 / 4.0
    assert recall == 2.0 / 3.0
    assert f1 == (2 * precision * recall) / (precision + recall)
    assert num_gold == 3
    assert num_pred == 4


def main():
    test_segmentation_evaluation()
    print("The tests passed.")


if __name__ == '__main__':
    main()