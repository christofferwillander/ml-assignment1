import csv as csv
import random as rand

def main():
    data = readDataFile("./data/spambase.data")
    spam = []
    ham = []
    trainingData = []

    for entry in data:
        if entry[57] == '1':
            spam.append(entry)
        else:
            ham.append(entry)
    
    # Shuffling the training data (positive class)
    rand.shuffle(spam)

    # Asserting 70 % of the positive class as training data
    trainingCount = int(len(spam)*0.7)
    trainingData = spam[:trainingCount]

    # Asserting 30 % of the positive class and the entire negative class as test data
    testData = spam[trainingCount:] + ham

    # Training model
    modelMin, modelMax = trainModel(trainingData)
    
    # Verifying model against test dataset
    TP, TN, FP, FN = verifyModel(testData, modelMin, modelMax)

    # Calculating performance metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    F1 = 2*(recall * precision) / (recall + precision)

    # Printing performance metrics
    print ("Accuracy: " + str(accuracy*100) + " %")
    print ("Precision: " + str(precision*100) + " %")
    print ("Recall: " + str(recall*100) + " %")
    print ("Specificity: " + str(specificity*100) + " %")
    print ("False-positive rate: " + str(FPR*100) + " %")
    print ("False-negative rate: " + str(FNR*100) + " %")
    print ("F1 score: " + str(F1))



def readDataFile(filePath):
    # Reading CSV data from the dataset
    file = open(filePath)
    csvReader = csv.reader(file)
    rows = []

    for row in csvReader:
        rows.append(row)

    return rows

def trainModel(trainingData):

    # Initializing
    modelLower = []
    modelLower = [float(100) for i in range(57)]
    modelUpper = []
    modelUpper = [float(0) for i in range(57)]
    
    for x in range(len(trainingData)):
        for y in range((len(trainingData[x]) - 1)):
            if float(trainingData[x][y]) < modelLower[y]:
                modelLower[y] = float(trainingData[x][y])
            elif float(trainingData[x][y]) > modelUpper[y]:
                modelUpper[y] = float(trainingData[x][y])


    return modelLower, modelUpper

def verifyModel(testData, modelLower, modelUpper):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for entry in testData:
        modelPrediction = checkEntry(entry, modelLower, modelUpper)

        if entry[-1] == '1':
            # Model predicted spam, it was spam
            if modelPrediction == 1:
                TP = TP + 1
            # Model predicted not spam, it was spam
            elif modelPrediction == 0:
                FN = FN + 1
        elif entry[-1] == '0':
            # Model predicted spam, it was not spam
            if modelPrediction == 1:
                FP = FP + 1
            # Model predicted not spam, it was not spam
            elif modelPrediction == 0:
                TN = TN + 1 

    return TP, TN, FP, FN

def checkEntry(entry, modelLower, modelUpper):
    for x in range((len(entry) - 1)):
        if (float(entry[x]) < modelLower[x] or float(entry[x]) > modelUpper[x]):
            return 0
    
    return 1

if __name__ == "__main__":
    main()