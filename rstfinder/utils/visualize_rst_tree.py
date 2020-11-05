#!/usr/bin/env python

"""
This script generates an html file with a tree visualization, using d3.js.
The template (template_visualize_rst_tree.html) is based off some D3.js
examples:

    http://mbostock.github.io/d3/talk/20111018/tree.html
    http://bl.ocks.org/mbostock/4063570
    http://www.d3noob.org/2013/01/adding-tooltips-to-d3js-graph.html

For the D3.js license, see ``LICENSE_d3.txt`` in ths directory or
https://github.com/mbostock/d3/blob/master/LICENSE for the D3.js license.

If you want to extract an SVG file from the output HTML file, use
this: http://nytimes.github.io/svg-crowbar/.

NOTE: On very large trees, edges may cross.  This appears to be a feature of
the layout algorithm.  It could be avoided by having curved edges, but
functionality for computing the necessary paths is likely not available.
"""

import argparse
import json
import os
from os.path import abspath, dirname, join

from jinja2 import Environment, FileSystemLoader

from nltk.tree import ParentedTree

THIS_FILE_DIRNAME = dirname(abspath(__file__))


def convert_tree_json(input_json):
    """Convert the JSON from the RST parser into a format for D3.js."""
    tree = ParentedTree.fromstring(input_json["scored_rst_trees"][0]["tree"])
    edus = [' '.join(x) for x in input_json["edu_tokens"]]
    res = _convert_tree_json_helper(tree, edus)
    return res


def _convert_tree_json_helper(subtree, edus):
    """Helper function for ``convert_tree_json()``."""
    if subtree.label() == "text":
        return {"name": edus[int(subtree[0])]}
    return {"name": subtree.label(),
            "children": [_convert_tree_json_helper(x, edus) for x in subtree]}


def main():  # noqa: D103
    """
    Main function.

    Args:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_json_path",
                        help="JSON file with output from the RST discourse parser.")
    parser.add_argument("output_html_path",
                        help="Path for the HTML output (note that ``d3.min.js`` "
                             "will need to be in the same directory when viewing "
                             "this file)")
    parser.add_argument("--embed_d3js",
                        help="If specified, the d3.min.js file will be "
                             "embedded in the output HTML for offline viewing. "
                             "By default, a link to ``cdnjs.cloudflare.com`` "
                             "will be included.",
                        action="store_true")
    args = parser.parse_args()

    if args.input_json_path == args.output_html_path:
        raise ValueError("The input and output paths are the same.")

    env = Environment(loader=FileSystemLoader(THIS_FILE_DIRNAME))
    tmpl_overview = env.get_template("template_visualize_rst_tree.html")

    with open(args.input_json_path) as inputfh:
        input_json = json.load(inputfh)
    tree_json = convert_tree_json(input_json)

    if args.embed_d3js:
        with open(os.path.join(THIS_FILE_DIRNAME, "d3.min.js")) as d3fh:
            d3_js = ('<script type="text/javascript">{}</script>'
                     .format(d3fh.read()))
    else:
        d3_js = '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.4.13/d3.min.js"></script>'

    html_output = tmpl_overview.render(tree_json=tree_json, d3_js=d3_js)

    with open(args.output_html_path, 'w') as outfile:
        print(html_output, file=outfile)


if __name__ == "__main__":
    main()
