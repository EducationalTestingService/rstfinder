import sys
import pydot
from pyparsing import OneOrMore, nestedExpr

### naive visualization of RST in png format
### run on Python 3
### usage: python viz_discourse.py test.dis
### result: example_graph.png, where each node has span and RST labels.
###         Edges with '*' indicates the Nucleus its parent.

f = open(sys.argv[1], 'r').read()
data = (OneOrMore(nestedExpr()).parseString(f))

def draw(parent_name, child_name, rst_type=None):
    if rst_type == 'Nucleus':
        edge = pydot.Edge(parent_name, child_name, label="*")
        graph.add_edge(edge)
    else:
        edge = pydot.Edge(parent_name, child_name)
        graph.add_edge(edge)

def dictToGraph(d, parent=None):
    if d['rst_type'] == 'Root':
        node_name = "-".join(d['Span'][1:])
        for child in d['children']:
            dictToGraph(child, node_name)

    elif 'text' in d:
        node_name = d['Span'] + '_' + d['rel2par']
        draw(parent, node_name, d['rst_type'])
        draw(node_name, d['text'], d['rst_type'])

    else:
        node_name = "-".join(d['Span'][1:]) + '_' + d['rel2par']
        flag = True
        for child in d['children']:
            if flag:
                draw(parent, node_name, d['rst_type'])
                flag = False
            else:
                pass
            dictToGraph(child, node_name)

def listToDict(l):
    node_type = l[0]
    if node_type == 'Root':
        node_span = l[1]
        node_children = l[2:]
        node_property = []
        for node_child in node_children:
            node_property.append(listToDict(node_child))
        final_dict = {'rst_type':node_type, 'children':node_property, 'Span':node_span}

    else:
        if l[1][0] == 'span':
            node_span = l[1]
            rel2par = l[2][1]
            node_children = l[3:]
            node_property =[]
            for node_child in node_children:
                node_property.append(listToDict(node_child))
            temp_dict = {'rst_type': node_type, 'children': node_property, 'rel2par':rel2par, 'Span':node_span}
            return temp_dict

        elif l[1][0] == 'leaf':
            node_span = str(l[1][1])
            rel2par = l[2][1]
            text = l[3]
            temp_dict = {}
            if text[0] == 'text':
                text=text[1:]
            if len(text) != 1:
                text = " ".join(str(x) for x in text)[2:-2]
            else:
                text = text[0][2:-2]+' ' # the last blank is to avoid an error by pydot

            temp_dict = {'rst_type': node_type, 'text': text, 'rel2par':l[2][1], 'Span':node_span}
            return temp_dict
    
    return final_dict
    
## for debugging purpose
#print(data[0])
#print(listToDict(data[0]))

# plot a graph
graph = pydot.Dot(graph_type='graph')
dictToGraph(listToDict(data[0]))
graph.write_png('example_graph.png')

