# ml-assignment1
**Assignment 1 in the course Machine Learning (DV2578). ML concept learner in Python for detecting spam e-mails.**

## General information
Model information, model performance metrics and the *LGG* (*LGG-Conj-ID*) rule for the training data (*positive class*) is printed in the terminal after execution of the *spam-classifier.py* script based on the *Spambase* data set from *UCI Machine Learning Repository*. The numbers in each array in the LGG rule corresponds to the bins for the feature in question in the training data set, i.e. bins that correspond to spam traits.

Training data is discretizised using *pandas*, where a total of 50 bins are used. The test data is discretizised using the bin bounds given by the training data, and hence they have the same scale. The data set is shuffled during each execution, to avoid bias.

*Avg. accurary:* ~82 %\
*Avg. precision:* ~47 %\
*Avg. recall:* ~79 %\
*Avg. specificity:* ~82 %\
*Avg. false-positive rate:* ~17 %\
*Avg. false-positive rate:* ~21 %\
*Avg. F1 score:* ~0.59

## Setup
**Requirements:** *Python 3.10.0*, *pip 21.3.1*, *pandas*

Simply install Python and pip, and then run the *pip install -r requirements.txt* command in the terminal to install *pandas*.
**N.B.** Make sure CWD is the root of the assignment folder.


## Script execution
The script is executed by simply typing *python spam-classifier.py* in the terminal while the CWD is the root of the assignment folder.

## Author
Christoffer Willander (DVACD17) - *chal17@student.bth.se*