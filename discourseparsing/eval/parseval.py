#!/usr/bin/env python

"""
Calculates parsing evaluation metrics: precision, recall, labeled precision and
labeled recall.
"""

from nltk.tree import *
import copy

data_floder_name = 'data/'
rule_file = data_floder_name+'Rule.txt'
test_file = data_floder_name+'test.txt'
test_sentence_file = data_floder_name+'testsentence.txt'
parsed_sentence_file = data_floder_name+'testParses.txt'
corpus_file = data_floder_name+'corpus.txt'
cnf_rule_file = data_floder_name+'ChomskyRule.txt'

def precision(gold, parse, ignore_labels=True):
    """Return the proportion of brackets in the suggested parse tree that are
    in the gold standard. Parameters gold and parse are NLTK Tree objects."""
    suc,total = precision_half(gold, parse, ignore_labels)
    return float(suc)/total

def precision_half(gold,parse,ignore_labels=True):
    
    parsebrackets = list_brackets(parse)
    goldbrackets = list_brackets(gold)

    parsebrackets_u = list_brackets(parse, ignore_labels=True)
    goldbrackets_u = list_brackets(gold, ignore_labels=True)
    print parse

    if ignore_labels:
        candidate = parsebrackets_u
        gold = goldbrackets_u
    else:
        candidate = parsebrackets
        gold = goldbrackets

    total = len(candidate)
    successes = 0
    for bracket in candidate:
        if bracket in gold:
            successes += 1
    return (successes,total)

def recall(gold, parse, ignore_labels=True):
    """Return the proportion of brackets in the gold standard that are in the
    suggested parse tree."""
    suc,total = recall_half(gold, parse, ignore_labels)
    return float(suc)/total

def recall_half(gold, parse, ignore_labels=True):
    parsebrackets = list_brackets(parse)
    goldbrackets = list_brackets(gold)

    parsebrackets_u = list_brackets(parse, ignore_labels=True)
    goldbrackets_u = list_brackets(gold, ignore_labels=True)

    if ignore_labels:
        candidate = parsebrackets_u
        gold = goldbrackets_u
    else:
        candidate = parsebrackets
        gold = goldbrackets

    total = len(gold)
    successes = 0
    for bracket in gold:
        if bracket in candidate:
            successes += 1
    return (successes,total)

def compute_fscore(precision, recall):
	if precision == 0:
		return 0
	if recall == 0:
		return 0

	return (2 * precision * recall) / (precision + recall)
	
def corpus_eval(goldfile,parsefile):
    
    paridx = 0
    goldidx = 0
    pcs_suc = 0
    pcs_tol = 0
    rcl_suc = 0
    rcl_tol = 0
    
    gold = open(goldfile).readlines()
    parse = open(parsefile).readlines()
    if len(gold)!= len(parse):
        print 'Error: Not the same size'
        return False
    while paridx < len(parse):
        if True:
            gold_tree = Tree.parse(gold[goldidx].decode('gbk'))
            parse_tree = Tree.parse(parse[paridx].decode('gbk'))
            
            s,t = precision_half(gold_tree,parse_tree)
            pcs_suc += s
            pcs_tol += t
            
            s,t = recall_half(gold_tree,parse_tree)
            rcl_suc += s
            rcl_tol += t
            
            paridx += 1
            goldidx += 1
        else:
            goldidx += 1

    pcs = float(pcs_suc)/(pcs_tol)
    rcl = float(rcl_suc)/(rcl_tol)
    
    print('Precision: %6.4f'%pcs)
    print('Recall:    %6.4f'%rcl)
    print('F-score:   %6.4f'%(2*pcs*rcl/(pcs+rcl)))
        
        
def labeled_precision(gold, parse):
    return precision(gold, parse, ignore_labels=False)

def labeled_recall(gold, parse):
    return recall(gold, parse, ignore_labels=False)

def words_to_indexes(tree):
    """Return a new tree based on the original tree, such that the leaf values
    are replaced by their indexs."""

    out = copy.deepcopy(tree)
    leaves = out.leaves()
    for index in range(0, len(leaves)):
        path = out.leaf_treeposition(index)
        out[path] = index + 1
    return out

def firstleaf(tr):
    return tr.leaves()[0]

def lastleaf(tr):
    return tr.leaves()[-1]

def list_brackets(tree, ignore_labels=False):
    tree = words_to_indexes(tree)

    def not_pos_tag(tr):
        return tr.height() > 2

    def label(tr):
        if ignore_labels:
            return "ignore"
        else:
            return tr.label()

    out = []
    subtrees = tree.subtrees(filter=not_pos_tag)
    return [(firstleaf(sub), lastleaf(sub), label(sub)) for sub in subtrees]

def example1():
    gold = Tree.parse(
"""
(PX
    (PX
        (APPR an)
        (NX
            (ART einem)
            (NX
                (NX (NN Samstag))
                (KON oder)
                (NX (NN Sonntag)))))
    (ADVX (ADV vielleicht)))
""")

    parse = Tree.parse(
"""
(PX
    (PX
        (APPR an)
        (NX
            (ART einem)
            (NN Samstag)))
    (NX (KON oder) (NX (NN Sonntag)))
    (ADVX (ADV vielleicht)))
""")
    
    pscore = precision(gold,parse)
    rscore = recall(gold,parse)
    fscore = compute_fscore(pscore, rscore)
    print pscore, rscore, fscore

def example2():
    gold = Tree.parse(
"""
(SIMPX
    (VF
        (PX (appr von)
            (NX
                (pidat allen)
                (ADJX (adja kulturellen))
                (nn Leuchttuermen))))
    (LK
        (VXFIN (vxfin besitzt)))
    (MF
        (ADVX
            (ADVX (adv nach))
            (kon wie)
            (ADVX (adv vor)))
        (NX
            (art das)
            (nn Theater))
        (NX
            (art das)
            (ADJX (adja unsicherste))
            (nn Fundament))))
""")

    parse = Tree.parse(
"""
(R-SIMPX
    (LV
        (PX
            (appr von)
            (NX (pidat allen))))
    (VF
        (NX
            (ADJX (adja kulturellen))
            (nn Leuchttuermen)))
    (LK (VXFIN (vvfin besitzt)))
    (MF
        (PX
            (PX (appr nach))
            (kon wie)
            (PX
                (appr vor)
                (NX
                    (art das)
                    (nn Theater))))
        (NX
            (art das)
            (ADJX (adja unsicherste))
            (nn Fundament))))
""")
    print "Precision:", precision(gold,parse)
    print "Labeled precision:", labeled_precision(gold,parse)
    print "Recall:", recall(gold,parse)
    print "Labeled recall:", labeled_recall(gold,parse)

def main():
    example1()
    #corpus_eval(test_file, parsed_sentence_file)
#    example2()
    
if __name__ == "__main__": main()