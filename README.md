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

## Running the final project
to run the 2 linear regression examples
`./main.py -f1 ~/tracks/track1.csv -f2 ~/tracks/track2.csv`
**note** adding argument -p will produce plots

Training the project model 
`./main.py  -d ../kalmanPy/tracks/`

Running the project model with pre-trained weights 
`./main.py  -d ../kalmanPy/tracks/`
