# ml-assignment1
**Assignment 1 in the course Machine Learning (DV2578). ML concept learner in Python for detecting spam e-mails.**

## General information
Model information, model performance metrics and the *LGG* (*LGG-conj*) rule is printed in the terminal after execution of the *spam-classifier.py* script based on the *Spambase* data set from *UCI Machine Learning Repository*.

Training data is discretizised using *pandas*, where a total of 100 bins are used. The test data is discretizised using the bin bounds given by the training data, and hence they have the same scale. The data set is shuffled during each execution, to avoid bias.

*Avg. accurary:* ~83 %\
*Avg. precision:* ~48 %\
*Avg. recall:* ~71 %\
*Avg. specificity:* ~85 %\
*Avg. false-positive rate:* ~15 %\
*Avg. false-positive rate:* ~28 %\
*Avg. F1 score:* ~0.58

## Setup
**Requirements:** *Python 3.10.0*, *pip 21.3.1*, *pandas*

Simply install Python and pip, and then run the *pip install -r requirements.txt* command in the terminal to install *pandas*.
**N.B.** Make sure CWD is the root of the assignment folder.


## Script execution
The script is executed by simply typing *python spam-classifier.py* in the terminal while the CWD is the root of the assignment folder.

## Author
Christoffer Willander (DVACD17) - *chal17@student.bth.se*