# Predicting Slow Judgments

## Motivation

Imagine you read this statement on a blog:

> San Juan city council votes unanimously to impeach Trump-hating mayor.

You can make a quick guess about whether this is true or false, but to really be sure most people would have to do additional research.

We are interested in problems like this, where (1) the true answer can only be obtained (with high confidence) through a relatively lengthy/expensive process that may involve thinking, gathering information, discussion, and more; but (2) quick guesses are possible and contain useful information. Specifically, we are interested in the task of predicting how human judgments to such questions evolve over time.

## Setup

You will need Anaconda installed, Python with development headers, and a C compiler installed. 

If you need to install and update anaconda, you can do so with:
```
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh
conda update --all
```

Then, you can install the repo using:

```
conda env create -f=environment.yml
```

Before running files in the repo, you should additionally load the [direnv](https://github.com/direnv/direnv) configuration file, which you can do with:

```
source .envrc
```

## Outline of the repository

The repository consists of some _datasets_ consisting of slow judgments to some questions, and some _models_ that predict slow judgments, and _a script_ for question-generation.

The datasets are available in `data`. The data including questions and human responses for the Fermi questions is in `data/human-results/fermi` and the corresponding data for Politifact questions is in `data/human-results/politifact`. This is what we expect people will want to use in order to generate their own models.

Our models are in `src/models`, and are demonstrated in `notebooks`.

The script for generating more Fermi questions is `src/data/generate_questions/generate_fermi_data.py`. These new questions don't include any probabilistic answers. In theory, additional Fermi questions could be used to improve the language processing parts of a model.


## Maintenance

If a package is added to the `environment.yml` file, then you must run:

```bash
conda env update -f environment.yml
```

from the `psj` environment to install the new packages.

## Tests

To run the tests, run

```
nosetests
```

