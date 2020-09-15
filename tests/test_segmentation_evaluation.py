from nose.tools import eq_
from rstfinder.tune_segmentation_model import convert_crfpp_output, evaluate_segmentation_output


def test_segmentation_evaluation():
    """Check that the segmentation evaluation works as expected."""
    output = ("foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n" "foo\tC-EDU\tC-EDU\n"
              "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tB-EDU\n" "foo\tC-EDU\tB-EDU\n"
              "foo\tC-EDU\tC-EDU\n" "foo\tC-EDU\tC-EDU\n" "\n"
              "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n" "foo\tB-EDU\tC-EDU\n"
              "foo\tC-EDU\tC-EDU\n" "foo\tB-EDU\tB-EDU\n" "foo\tC-EDU\tC-EDU\n")
    converted = convert_crfpp_output(output)
    (precision,
     recall,
     f1,
     num_gold,
     num_pred) = evaluate_segmentation_output(converted)

    eq_(precision, 2.0 / 4.0)
    eq_(recall, 2.0 / 3.0)
    eq_(f1, (2 * precision * recall) / (precision + recall))
    eq_(num_gold, 3)
    eq_(num_pred, 4)
