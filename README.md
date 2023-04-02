# pyDeepLearn Module
## Description
This python module was developed to investigate the ins and outs of a deep learning framework by building my own.
## install 
This module was developed using python 3.8.10, for additional dependancy list see the included requirements.txt file

* install python 3.8.10
* (optional) create a python virtual env to seperate current system deps from deps required for this package 

  `python3 -m pip install virtualenv`

  `mkdir python-venv`

  `python3 -m venv python-venv`

  `source python-venv/bin/activate`

* install required dependancies

  `pip install -r requirements.txt`

  **Note:** venv and dep install can also be completed by running `make env`
## Creating installable whl package
This module can be built into an installable python whl package but running
the following command from the repo root

`make pkg`

Once this has been created the package can be installed using

`python3 -m pip install dist/pyDeepLearn-1.0-py2.py3-none-any.whl`

## Running unit tests
### From the directory root 
`make check`

### From the installed package
`python -m pyDeepLearn.tests.run`

## Running HW 1 specific instructions
The following instructions provide you with details to run the homework problems as defined in questions 5, 6, and 7

This is intended to be ran from the repo root directory 

`./hw1/hw1_main.py -f <path to CSV data> -v `

For example 

`./hw1/hw1_main.py -f ~/mcpd_augmented.csv -v`

## Running HW 2 specific instructions
The following instructions provide you with details to run the homework problems as defined in questions 1, 4, and 5

This is intended to be ran from the repo root directory 

`./hw2/hw2_main.py `

## Running HW3 specific instructions
The following instruction set provide you with information to recreate the requested plots from problems 1, 4, and 5. As well 
as calculate the accuracy of problem 5

`./hw3/hw3_main.py -f1 hw3/mcpd_augmented1.csv  -f2 hw3/KidCreative.csv`

**Note:** f1 is used to point to the mcpd data you have stored localy, and f2 points to Kid creative. The path above was provided only as an example and will not be included with the repo.

## Running HW4 specific instructions
The following instruction set provide you with information to recreate the requested plots

`./hw4/hw4_main.py -f1  ~/mnist_short_train.csv  -f2 ~/mnist_short_test.csv`

## Running HW5 specific instructions
`./hw5/hw5_main.py -f1 ~/mnist_train_100.csv -f2 ~/mnist_valid_10.csv`

**Note:** f1 is used to point to the mnist training data you have stored localy, and f2 points to mnist test data. The path above was provided only as an example and will not be included with the repo.

## Running the final project
to run the 2 linear regression examples
`./main.py -f1 ~/tracks/track1.csv -f2 ~/tracks/track2.csv`
**note** adding argument -p will produce plots

Training the project model 
`./main.py  -d ../kalmanPy/tracks/`

Running the project model with pre-trained weights 
`./main.py  -d ../kalmanPy/tracks/`
