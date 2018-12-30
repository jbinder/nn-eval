nn-eval
=======

Evaluates different neural network libraries on simple datasets.

For unspecified training and network options it will try to determine the best values.

Currently only PyTorch is supported.


Requirements
------------

* Python 3.6
* Conda 4.5


Install
-------

        conda env create


Usage
-----

Activate the environment:

        conda activate env-nn-eval

Run the app:

        python main.py

The app gets the input from two CSV files, where one (x.csv) contains the input, and the other (y.csv) the expected output.
The first column of each CSV is treated as identity column.
Only rows where the identity exists in both CSV files are included.
Validation data is read in the same format from x_valid.csv and y_valid.csv.
For examples, see the [demo data](data).


Development
-----------

To update the environment after changed dependencies, run:

        conda env update
