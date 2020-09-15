#!/bin/zsh

# set up some paths
CURRDIR=$(pwd)
CONDA_ENV_PATH="${CURRDIR}"/../parserdev

# remove the conda environment if it already exists
[[ -d "${CONDA_ENV_PATH}" ]] && /opt/python/conda_default/bin/conda env remove -p "${CONDA_ENV_PATH}"

# create conda environment
CONDA_ENV_PATH="${CURRDIR}"/../parserdev
/opt/python/conda_default/bin/conda create --override-channels -c conda-forge -c ets -c https://nlp.research.ets.org/conda -p "${CONDA_ENV_PATH}" python=3.6 nose crf++ --file conda_requirements.txt

# install the bookadmin package in development mode
"${CONDA_ENV_PATH}"/bin/pip install -e .

# run the tests and generate the coverage report
"${CONDA_ENV_PATH}"/bin/nosetests --nologcapture tests --with-cov --cov rstfinder --cov-report=html --with-xunit
