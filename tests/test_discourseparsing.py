import json
from pathlib import Path

from nltk.tree import ParentedTree
from nose import SkipTest
from nose.tools import eq_
from rstfinder.discourse_parsing import Parser
from rstfinder.extract_actions_from_trees import extract_parse_actions

MY_DIR = Path(__file__).parent


def test_extract_parse_actions():
    """Check that parse actions are extracted as expected."""
    # the following tree represents a sentence like:
    # "John said that if Bob bought this excellent book,
    # then before the end of next week Bob would finish it,
    # and therefore he would be happy."
    tree = ParentedTree.fromstring("""(ROOT
                                      (satellite:attribution (text 0))
                                      (nucleus:span
                                          (satellite:condition (text 1))
                                          (nucleus:span
                                              (nucleus:span
                                                  (nucleus:same-unit (text 2))
                                                  (nucleus:same-unit
                                                      (satellite:temporal (text 3))
                                                      (nucleus:span (text 4))))
                                              (satellite:conclusion (text 5)))))
                                    """)
    actions = extract_parse_actions(tree)
    num_shifts = len([x for x in actions if x.type == 'S'])
    eq_(num_shifts, 6)
    eq_(actions[0].type, 'S')
    eq_(actions[1].type, 'U')
    eq_(actions[1].label, "satellite:attribution")
    eq_(actions[2].type, 'S')


def test_reconstruct_training_examples():
    """Check extracted actions for entire training data."""
    # go through the training data and make sure
    # that the actions extracted from the trees can be used to
    # reconstruct those trees from a list of EDUs

    # check if the training data file exists, otherwise skip test
    file_path = Path('rst_discourse_tb_edus_TRAINING_TRAIN.json')
    if not file_path.exists():
        raise SkipTest("training data JSON file not found")

    # read in the training data file
    with open(file_path) as train_data_file:
        data = json.load(train_data_file)

    # instantiate the parser
    rst_parser = Parser(max_acts=1, max_states=1, n_best=1)

    # iterate over each document in the training data
    for doc_dict in data:

        # get the original RST tree
        original_tree = ParentedTree.fromstring(doc_dict['rst_tree'])

        # extract the parser actions from this tree
        actions = extract_parse_actions(original_tree)

        # reconstruct the tree from these actions using the parser
        reconstructed_tree = next(rst_parser.parse(doc_dict,
                                                   gold_actions=actions,
                                                   make_features=False))['tree']

        eq_(reconstructed_tree, original_tree)


def test_parse_single_edu():
    """Test that parsing a single EDU works as expected."""
    # load in an input file with a single EDU
    input_file = MY_DIR / "data" / "single_edu_parse_input.json"
    doc_dict = json.load(open(input_file, 'r'))

    # instantiate the parser
    rst_parser = Parser(max_acts=1, max_states=1, n_best=1)
    rst_parser.load_model(str(MY_DIR / "models"))

    # make sure that we have a very simple RST tree as expected
    tree = list(rst_parser.parse(doc_dict))[0]['tree']
    eq_(tree.root().label(), 'ROOT')
    eq_(len(tree.leaves()), 1)
    subtrees = list(tree.subtrees())
    eq_(len(subtrees), 2)
