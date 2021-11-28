import pandas as pd
import numpy as np

def main():
    # Reading dataset into Pandas DataFrame
    data = pd.read_csv("./data/spambase.data", header=None)

    # Selecting the spam data
    spam = data.loc[data[57] == 1]

    # Shuffling the spam data
    spam = spam.sample(frac = 1)

    # Selecting the ham data
    ham = data.loc[data[57] == 0]

    # Asserting 70 % of the positive class as training data
    trainingCount = int(len(spam)*0.7)
    trainingData = spam[:trainingCount]

    # Dropping the boolean spam parameter from training data set
    trainingData = trainingData.iloc[:, : -1]

    # Asserting 30 % of the positive class and the entire negative class (ham) as test data
    frames = [spam[trainingCount:], ham]
    testData = pd.concat(frames)
    testData.reset_index(inplace = True, drop = True)


    # Discretizising all training data features into 8 equal-width bins (creating model)
    trainingBins = []
    binnedTrainingData = trainingData.copy()
    nrOfBins = 100
    labels = [i + 1 for i in range(nrOfBins)]
    print("------ Model information ------")
    print("Number of bins: " + str(nrOfBins))

    for i in range(len(trainingData.columns)):
        trainingBins.append(0)
        binnedTrainingData[i], trainingBins[i] = pd.cut(x=binnedTrainingData[i], bins=nrOfBins, ordered=True, duplicates="drop", include_lowest=True, labels=labels, retbins=True)

    # Discretizising all test data features into 8 equal-width bins (based on training data ranges)
    for i in range(len(trainingData.columns - 1)):
        testData[i] = pd.cut(x=testData[i], bins=trainingBins[i], ordered=True, duplicates="drop", include_lowest=True, labels=labels)
    
    # Calculating hypothesis space based on the discretizised training data
    hypothesisSpace = calcHypothesisSpace(binnedTrainingData, nrOfBins)
    print("Size of hypothesis space: " + str(hypothesisSpace))

    # Calculating conjunctive concepts based on the discretizised training data
    conjunctiveConcepts = calcConjunctiveConcepts(binnedTrainingData, nrOfBins)
    print("Number of conjunctive concepts: " + str(conjunctiveConcepts))

    # Verifying model against test dataset
    TP, TN, FP, FN = verifyModel(testData, binnedTrainingData)

    # Calculating performance metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    F1 = 2*(recall * precision) / (recall + precision)

    # Printing performance metrics
    print("\n------ Performance metrics ------")
    print ("Accuracy: " + str(round(accuracy*100, 2)) + " %")
    print ("Precision: " + str(round(precision*100, 2)) + " %")
    print ("Recall: " + str(round(recall*100, 2)) + " %")
    print ("Specificity: " + str(round(specificity*100,2)) + " %")
    print ("False-positive rate: " + str(round(FPR*100, 2)) + " %")
    print ("False-negative rate: " + str(round(FNR*100, 2)) + " %")
    print ("F1 score: " + str(round(F1, 2)))

    # Printing LGG rule
    conjunctiveRule = LGG(testData, binnedTrainingData)
    print("\n------ Least General Generalization (LGG) rule ------")
    for feature in range(len(conjunctiveRule)):
        print("Feature " + str(feature + 1) + ": " + str(conjunctiveRule[feature]))

def calcHypothesisSpace(binnedTrainingData, nrOfBins):
    binCounts = [0 for i in range(nrOfBins)]
    hypothesisSpace = 1

    # Calculating number of non-empty bins for each feature
    for feature in range(len(binnedTrainingData.columns)):
        uniqueBins = binnedTrainingData[feature].nunique()
        binCounts[uniqueBins - 1] = binCounts[uniqueBins - 1] + 1

    # Calculating hypothesis space based on total bin counts
    for x in range(nrOfBins):
        hypothesisSpace = hypothesisSpace * pow((x + 1), binCounts[x])

    return hypothesisSpace

def calcConjunctiveConcepts(binnedTrainingData, nrOfBins):
    binCounts = [0 for i in range(nrOfBins + 1)]
    conjunctiveConcepts = 1

    # Calculating number of non-empty bins for each feature
    for feature in range(len(binnedTrainingData.columns)):
        uniqueBins = binnedTrainingData[feature].nunique()
        binCounts[uniqueBins] = binCounts[uniqueBins] + 1

    # Calculating conjunctive convepts based on total bin counts
    for x in range(nrOfBins):
        conjunctiveConcepts = conjunctiveConcepts * pow((x + 1), binCounts[x])

    return conjunctiveConcepts

def verifyModel(binnedTestData, binnedTrainingData):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    uniqueBins = []

    # Extracting the unique bins for each feature (i.e. spam traits)
    for feature in range(len(binnedTrainingData.columns)):
        bins = binnedTrainingData[feature].unique()
        uniqueBins.append(bins)

    # Iterating through discretizised test data to evaluate model
    for index, row in binnedTestData.iterrows():
        modelPrediction = checkEntry(row, uniqueBins)

        if row[len(row) - 1] == 1:
            # Model predicted spam, it was spam
            if modelPrediction == 1:
                TP = TP + 1
            # Model predicted not spam, it was spam
            elif modelPrediction == 0:
                FN = FN + 1
        elif row[len(row) - 1] == 0:
            # Model predicted spam, it was not spam
            if modelPrediction == 1:
                FP = FP + 1
            # Model predicted not spam, it was not spam
            elif modelPrediction == 0:
                TN = TN + 1 

    return TP, TN, FP, FN

def checkEntry(row, uniqueBins):
    for feature in range(len(row) - 1):
        if not row[feature] in uniqueBins[feature]:
            return 0
    return 1

def LGG(binnedTestData, binnedTrainingData):
    conjunctiveRule = []
    testDataBins = []
    trainingDataBins = []

    # Extracting unique bins for each feature in test data
    for feature in range(len(binnedTestData.columns)):
        testDataBins.append(binnedTestData[feature].unique())
    
    # Extracting unique bins for each feature in training data
    for feature in range(len(binnedTrainingData.columns)):
        trainingDataBins.append(binnedTrainingData[feature].unique())

    # For each feature, checking for common bins in training and test data sets
    for feature in range(len(trainingDataBins)):    
        bins = []

        for bin in range(len(testDataBins[feature])):
            if testDataBins[feature][bin] in trainingDataBins[feature]:
                bins.append(testDataBins[feature][bin])
        
        conjunctiveRule.append(bins)
    
    for i in range(len(conjunctiveRule)):
        conjunctiveRule[i].sort()
        
    return conjunctiveRule


if __name__ == "__main__":
    main()
