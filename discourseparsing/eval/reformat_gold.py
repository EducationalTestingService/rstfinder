import argparse
import logging
from nltk import ParentedTree



def delete_span_leaf_nodes(tree):
    subtrees = []
    subtrees.extend([ s for s in tree.subtrees() if s!= tree and (s.label() == 'span' or s.label() == 'leaf') ])
     
    if(len(subtrees) > 0):  
        parent = subtrees[0].parent()
        parent.remove(subtrees[0])
        delete_span_leaf_nodes(tree)

def move_rel2par(tree):
    subtrees = []
    subtrees.extend([ s for s in tree.subtrees() if s!= tree and (s.label() == 'rel2par') ])
     
    if(len(subtrees) > 0): 
        #there should only be one word describing the rel2par
        relation = ' '.join(subtrees[0].leaves())
        parent = subtrees[0].parent()
        #rename the parent node
        parent.set_label('{}:{}'.format(parent.label(), relation))

        #and then delete the rel2par node
        parent.remove(subtrees[0])
        move_rel2par(tree)


def reformat_trees(inputfile, outputfile):
    with open(inputfile, 'r') as reader:
        input_tree = ParentedTree(reader.read())
        logging.debug('Reformatting {}'.format(input_tree))

        #1. rename the top node
        for s in input_tree.subtrees():
            #directly modify the tree
            if s.label() == 'Root':
                s.set_label('TOP')

        #2. delete all of the span and leaf nodes (they seem to be just for book keeping)
        delete_span_leaf_nodes(input_tree)

        #3. move the rel2par label up to be attached to the Nucleus/Satellite node
        move_rel2par(input_tree)


        logging.debug('Reformatted: {}'.format(input_tree))





def main(arguments=[]):
    parser = argparse.ArgumentParser(description="Converts the gold standard rst parses in the rst treebank to look more like what the parser produces",
                                    conflict_handler='resolve', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--inputfile', help='Input gold standard rst parse from treebank', type=str, required=True)
    parser.add_argument('-o', '--outputfile', help='Output file containing reformated tree', type=str, required=True)


    
    args = parser.parse_args(*arguments)
    # initialize the loggers
    logging.basicConfig(level=logging.DEBUG)

    reformat_trees(args.inputfile, args.outputfile)



if __name__ == '__main__':
    main()
