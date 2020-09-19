#!/bin/zsh

# set up the conda path
set -x
if [[ -f conda_path ]]; then
    CONDA_ENV_PATH=$(cat conda_path)
fi || exit 1

# set up ZPar and NLTK environment variables
export NLPTOOLS="/home/nlp-text/dynamic/NLPTools"
export ZPAR_MODEL_DIR="${NLPTOOLS}/zpar/models/english"
export NLTK_DATA="${NLPTOOLS}/nltk_data"
export CORPORA="/home/nlp-text/static/corpora"

# create the JSON files from the RST and Penn treebanks
"${CONDA_ENV_PATH}"/bin/convert_rst_discourse_tb $CORPORA/nonets/rst_discourse_treebank/original/rst_discourse_treebank $CORPORA/nonets/treebank3/original/treebank_3 >& convert.log

# split train JSON into train + dev
"${CONDA_ENV_PATH}"/bin/make_traindev_split

# extract segmentation features for train and dev
"${CONDA_ENV_PATH}"/bin/extract_segmentation_features rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv
"${CONDA_ENV_PATH}"/bin/extract_segmentation_features rst_discourse_tb_edus_TRAINING_DEV.json rst_discourse_tb_edus_features_TRAINING_DEV.tsv

# train the segmentation model
"${CONDA_ENV_PATH}"/bin/tune_segmentation_model rst_discourse_tb_edus_features_TRAINING_TRAIN.tsv rst_discourse_tb_edus_features_TRAINING_DEV.tsv segmentation_model >& tune_segmenter.log

# save the best segmenter output
tail -6 tune_segmenter.log > best_segmenter_f1.txt

# train the RST parser
"${CONDA_ENV_PATH}"/bin/tune_rst_parser rst_discourse_tb_edus_TRAINING_TRAIN.json rst_discourse_tb_edus_TRAINING_DEV.json rst_parsing_model >& tune_rst_parser.log

# get best F1 value and check that it is within expected limits
F1VALUE=$(tail -1 tune_rst_parser.log | grep -o '0\.[0-9]\+[^$]' | sed 's/,//')
echo "$F1VALUE" > best_rst_parser_f1.txt
echo "${F1VALUE} > 0.58 && ${F1VALUE} < 0.60" | bc -l

# run any of the trained models on a test document
# NOTE: it doesn't matter which ones we run since
# we just want to make sure that the `rst_parse` command runs
"${CONDA_ENV_PATH}"/bin/rst_parse -g segmentation_model.C8.0 -p rst_parsing_model.C0.5 tests/data/rst_document.txt > output.json.txt

# run the visualizer next
visualize_rst_tree output.json tree.html --embed_d3js