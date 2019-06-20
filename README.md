nn-eval
=======

Evaluates different neural network libraries on simple datasets.

For unspecified training and network options it will try to determine the best values.

Currently PyTorch and Keras are supported.


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
By default, 70% of the data is used for training, 30% for validation.
For examples, see the [demo data](data).

Arguments:

-   **x** \<str>\: The path to the CSV file which contains input data (mandatory, default: x.csv).
-   **y** \<str>\: The path to the CSV file which contains the output data (mandatory, default: y.csv).
-   **data_train_percentage** \<float>\: The amount of data that should be used for training the network, the remaining data is used for validation  (mandatory, default: 0.7).
-   **size_hidden** \<int\> \<int\> ...: The list of hidden layer sizes to use (optional).
-   **gpu** \<True/False\>: Set to True to allow using the GPU if available (optional, default: True).
-   **optimizer** \<str\>: The optimization algorithm to use (optional).
-   **activation_function** \<str\>: The activation function to use (optional, default: relu).
-   **loss_function** \<str\>: The loss function to use (optional).
-   **dropout_rate** \<float>\: The dropout rate to be used for training the network (optional, default: 0.5).
-   **bias** \<True/False\>: Set to True to use a bias (optional, default: True).
-   **epochs** \<int\>: The number of epochs to use (optional).
-   **seed** \<int\>: Overrides the default random seed with a fixed custom one (optional).
-   **deterministic** \<True/False\>: Set to True to use a deterministic behavior, i.e. a fixed seed and no shuffling of the training data (optional, default: False).
-   **print_every** \<int\>: The interval in which log messages are shown (optional, default: 64).
-   **model_file** \<str\>: The path to where to best found model should be stored to (optional).
-   **batch_size** \<int\>: The batch size in which training should be performed (optional).
-   **num_runs_per_setting** \<int\>: To consider the random seed some libraries provide, run multiple times with the same settings (optional, default: 10).
-   **visualize** \<True/False\>: Set to True to show a plot of expected vs predicted values (optional, default: True).
-   **visualize_limit** \<int\>: See visualize, limit the plot to the specified number of values (optional).
-   **visualize_include_test_data** \<True/False\>: See visualize, set to True if not only validation but also test data should be shown (optional, default: False).
-   **progress_detection_patience** \<int\>: The number of epochs with little loss improvement until stopping to lear (optional, default: 1000).
-   **normalizer** \<str\>: The normalizer to use for normalizing the data. Currently only the reciprocal and identity normalizers are available (optional, default: Identity).

Example:

        # Try finding the best options for data in x.csv/y.csv using a hidden layer [8, 8]
        python main.py --x x.csv --y y.csv --size_hidden 8 8


Development
-----------

To update the environment after changed dependencies, run:

        conda env update
