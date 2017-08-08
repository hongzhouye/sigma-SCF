[![Build Status](https://travis-ci.org/hongzhouye/sigma-SCF.svg?branch=master)](https://travis-ci.org/hongzhouye/sigma-SCF)
[![codecov](https://codecov.io/gh/hongzhouye/sigma-SCF/branch/master/graph/badge.svg)](https://codecov.io/gh/hongzhouye/sigma-SCF)


# sigma-SCF
A direct energy targeting method to mean-field excited states
---

### Method

See our paper (coming soon).

### Install

1. First make sure `conda` is installed:
  * Mac instruction:
  ```bash
  curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O
  bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/local/opt/miniconda
  echo PATH="\$HOME/local/opt/miniconda/bin:\$PATH" >> ~/.bash_profile
  ```
  * Linux instruction:
  ```bash
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/local/opt/miniconda
  echo PATH="\$HOME/local/opt/miniconda/bin:\$PATH" >> ~/.bashrc
  ```
  * Windows instruction:
  ```
  You are on your own. GOOD LUCK!
  ```
2. Then create a conda virtual environment
  ```bash
  conda create -n sscf python=3.5 pytest pytest-cov pyyaml psi4 numpy lawrap gnu cmake scipy numexpr mkl-include gcc-5-mp -c intel -c psi4/label/dev -c psi4
  pip install codecov
  ```
  The above command (first line) creates a conda virtual environment with name `sscf` (for sigma-SCF), and installs necessary packages. At this point, if you type 
  ```bash
  source activate sscf
  ``` 
to activate the sigma-SCF virtual environment, by 
  ```bash
  echo ${CONDA_PREFIX}
  ```
you can see something like
  ```bash
  /Users/[user_name]/local/opt/miniconda/envs/sscf      # on a Mac
  ```
or
  ```bash
  /home/[user_name]/local/opt/miniconda/envs/sscf       # on a Linux computer
  ```
  
3. Then clone this repository
  ```bash
  git clone http://github.com/hongzhouye/sigma-SCF.git
  ```

4. In directory `sigma-SCF`
  ```bash
  source activate sscf          # activate virtual environment for sigma-SCF
  pip install -e .
  cd jk; cmake .; make; cd ..
  cd xform; cmake .; make; cd ..
  ```

5. Open a python interpreter, say `ipython`, and type
  ```python
  >>> import SuperCoolFast as scf
  ```
you should see no error messages.

### Tests

There are some built-in tests in directory `sigma-SCF/tests`. You can run from directory `sigma-SCF` by
  ```bash
  py.test -v
  ```

### Author Lists

#### Hongzhou Ye
Hi! This is Hongzhou from MIT. I am doing method development in electronic structure theory. Specifically we are developing methods for molecular systems w/ strong correlation.

I often code in C++. But I've been coding in python more and more since very recently.

#### Nadav Geva
Hi! I'm Nadav Geva from MIT. I work on non-local functionals, energy transferbetween QD and OSC, and other accronyms.

I ususlly program with a mix of python and C++.

#### Courtney Fitzgerald
Hi this is Courtney. I'm still very new to coding. I have a little experience with python. Still getting used to Bash and GitHub. Through this project, I'm hoping to learn a lot.

#### Shannon Houck
Hello! My name is Shannon Houck, and I'm from Virginia Tech.
I work with methods development, specifically focusing on embedding theory.
My primary language is Python, but I like C++ a lot, too!
