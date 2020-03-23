# Earthquake Predictor Neural Network

## Installation

The following dependencies should be installed:

* numpy
* pillow
* tensorflow >= 2.1

## Setup

Please create the following sub-directories:

* data
* model
* prediction
 * high-recall
 * high-precision
* test

## Downloading Data, Training, and Testing

Before downloading the data, make sure you have Internet access.  This script will download directly from the U.S. Geological Survey website earthquake data from 1973 to the present.  Then it will train models and test them on the year 2018.  Then it will further train the models to the present day.

Run the command:  `python earthquake_rnn.py`

## Updating The Website

Create a config.py file with the following string fields:

* ftp_url
* username
* password
* web_dir

Then run the command: `python updater.py`

## Licence

Apache 2.0