#!/bin/zsh

# set up the conda paths
CURRDIR=$(pwd)
CONDA_ENV_PATH="${CURRDIR}"/../parserdev

# set up ZPar and NLTK environment variables
export NLPTOOLS="/home/nlp-text/dynamic/NLPTools"
export ZPAR_MODEL_DIR="${NLPTOOLS}/zpar/models/english"
export NLTK_DATA="${NLPTOOLS}/nltk_data"
export ENVFILE="bamboo_ci/ci_env.yaml"

# remove the conda environment if it already exists
[[ -d "${CONDA_ENV_PATH}" ]] && /opt/python/conda_default/bin/conda env remove -p "${CONDA_ENV_PATH}"

# create the conda environment using the environment.yaml file
/opt/python/conda_default/bin/conda env create -f "${ENVFILE}" -p "${CONDA_ENV_PATH}" -v

# install the rstfinder package in development mode
"${CONDA_ENV_PATH}"/bin/pip install -e .

# link in the training JSON file that we need for the tests
ln -s /home/nlp-text/dynamic/mheilman/discourse-parsing/rst_discourse_tb_edus_TRAINING_TRAIN.json .

# run the tests and generate the coverage report
"${CONDA_ENV_PATH}"/bin/nosetests --nologcapture --verbose tests --with-coverage --cover-package rstfinder --cover-html --with-xunit

# save the conda environment path 
echo "${CONDA_ENV_PATH}" > conda_path
