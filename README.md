## COSC 6342 Course Project
### Binoy Dalal (1794070), Spring 2019
#### Gender classification of blog authors
***

Environment:
* Python 3.6
* Dependencies are present in the file requirements.txt. In order to install the dependencies, run `pip3 install -r requirements.txt`
* It is strongly suggested that the code be run on a system with Nvidia GPU support, as otherwise POS tag generation and tensorflow make take a long time to run
* _Note: Tensorflow-GPU requires CUDA and preferably cuDNN support to run effectively. These libraries need to be installed separately if not already present on the system. Alternatively, one can install the CPU version of tensorflow._


The code is self contained once the dependencies have been installed. To run the code, simply run `python3 main.py` from the directory of the source code. Otherwise there may be issues with loading the feature files.

All feature extraction code is present in `feature_extraction.py`

The main file runs 10-fold CV and test set evaluations on all features using MLP and XGBoost, Doc2Vec using MLP and SVM and RNN using Keras and tensorflow.

Depending on the available hardware configuration, the code can take more than 3 hrs. to complete a run. All results are displayed on stdout.

The `data` folder consists of the feature file, the csv data, and the word factor csv files.

The `notebooks` folder consists more than 20 jupyter notebooks that I used to prototype my solution and try out different approaches.

The `results` folder contains a spreadsheet which has recorded the results for various runs as presented in the paper. 
