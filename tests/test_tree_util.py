from nltk.tree import ParentedTree
from nose.tools import assert_raises, eq_
from rstfinder.tree_util import collapse_binarized_nodes, find_first_common_ancestor

EXAMPLE_TREE = ParentedTree.fromstring("""
                               (S
                                 (NP (PRP I))
                                 (VP
                                   (VBP am)
                                   (VP
                                     (VBG going)
                                     (PP (TO to) (NP (DT the) (NN market)))
                                     (PP (IN with) (NP (PRP$ her) (NN tomorrow)))))
                                 (. .))
                """)


def test_find_first_common_ancestor():
    """Test function to find first common ancestor."""
    # check that first common ancestor for (PP1, PP2) = VP
    pp1 = EXAMPLE_TREE[1][1][1]
    pp2 = EXAMPLE_TREE[1][1][2]
    computed_ancestor = find_first_common_ancestor(pp1, pp2)
    expected_ancestor = EXAMPLE_TREE[1][1]
    eq_(computed_ancestor, expected_ancestor)

    # check that first common ancestor for (VBG, NP) = S
    np = EXAMPLE_TREE[0]
    vbg = EXAMPLE_TREE[1][1][0]
    computed_ancestor = find_first_common_ancestor(vbg, np)
    expected_ancestor = EXAMPLE_TREE.root()
    eq_(computed_ancestor, expected_ancestor)


def test_find_first_common_ancestor_order():
    """Test that first common ancestor is order invariant."""
    np = EXAMPLE_TREE[0]
    vbg = EXAMPLE_TREE[1][1][0]
    computed_ancestor1 = find_first_common_ancestor(vbg, np)
    computed_ancestor2 = find_first_common_ancestor(np, vbg)
    eq_(computed_ancestor1, computed_ancestor2)


def test_find_first_common_ancestor_separate_trees():
    """Test that first common ancestor only works within a tree."""
    another_tree = ParentedTree.fromstring("(S (NP (PRP I)) (VP (VBP am) (ADVP (RB here))) (. .))")
    np_tree1 = EXAMPLE_TREE[0]
    np_tree2 = another_tree[0]
    assert_raises(AssertionError, find_first_common_ancestor, np_tree1, np_tree2)


def test_collapse_binarized_nodes():
    """Test that binarized nodes are properly collapsed."""
    # this a tree that's the same as the example tree but
    # with two temporary binarized nodes inserted
    tree_with_tmp_nodes = ParentedTree.fromstring("""
                                           (S
                                             (NP (PRP I))
                                             (VP
                                               (VBP am)
                                               (VP
                                                 (VBG going)
                                                 (PP (TO to) (NP (NP* (DT the) (NN market))))
                                                 (PP (IN with) (NP (NP* (PRP$ her) (NN tomorrow))))))
                                             (. .))
                           """)
    collapse_binarized_nodes(tree_with_tmp_nodes)
    eq_(tree_with_tmp_nodes, EXAMPLE_TREE)


def test_collapse_binarized_nodes_bad_label():
    """Test that binarized nodes with bad labels raise exception."""
    # this a tree that's the same as the example tree but
    # with a binarized node that has a different label
    # than its parent
    tree_with_tmp_nodes = ParentedTree.fromstring("""
                                           (S
                                             (NP (PRP I))
                                             (VP
                                               (VBP am)
                                               (VP
                                                 (VBG going)
                                                 (PP (TO to) (NP (TMP* (DT the) (NN market))))
                                                 (PP (IN with) (NP (NP* (PRP$ her) (NN tomorrow))))))
                                             (. .))
                           """)
    assert_raises(AssertionError, collapse_binarized_nodes, tree_with_tmp_nodes)
