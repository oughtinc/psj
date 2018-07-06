#!/usr/bin/env bash

set -ex

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
rm ./miniconda.sh
echo 'export PATH="~/miniconda2/bin:$PATH"' >> ~/.bashrc
export PATH="~/miniconda2/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update --yes conda
