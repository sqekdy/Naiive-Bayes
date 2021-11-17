# Author:  Sandip Gautam
#                           @sqekdy on github

# Implementation of Naive Bayes classification algorithm for numeric as well as categorical attribute
# To get help with running this module, execute  [ python classifier.py --help ]


import argparse
import pandas
import numpy as np
import math
import os


def gaussian(x, m, stdn):
    """
    :param x:   Value of Attribute Ak, for tuple X
    :param m:   Mean of the values of Attribute Ak, for a dataset D
    :param stdn:  Standard deviation of the values of Attribute Ak, for a dataset D

    :return:    Probabilities computed using the Gaussian Distribution Approach for given value of  attribute Ak
    """

    res = (1 / math.sqrt(2 * math.pi * stdn)) * math.exp(- ((x - m) ** 2) / (2 * (stdn ** 2)))

    return res


def evaluate_performance(original_data, predicted_data):
    """

    :param original_data:   Path to the file, where the original class labels are presenet
    :param predicted_data:  A list of predicted class labels for each of the tuples in the original data.
    :return: None, Output to stdout
    """

    actual_class = (pandas.read_table(original_data, sep=" ", header=None)).iloc[:, -1]

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(predicted_data)):

        if predicted_data[i] == actual_class[i]:

            if predicted_data[i] > 0:

                tp += 1

            elif predicted_data[i] < 0:

                tn += 1

        else:

            if predicted_data[i] > 0 and actual_class[i] < 0:

                fp += 1

            elif predicted_data[i] < 0 and actual_class[i] > 0:

                fn += 1

    confusion_matrix = pandas.DataFrame(np.array([[tp, fn], [fp, tn]]), columns=['1', '-1'], index=['1', '-1'])

    classsification_accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * tp) / ((2 * tp) + fp + fn)

    print("\nResult of Naive Bayes Classification on the %(d)s dataset:\n" % {'d':
                                                                                  os.path.split(
                                                                                      os.path.abspath(original_data))[
                                                                                      -1]})

    print("Confusion Matrix: ", "\n", confusion_matrix)
    print("Classification accuracy: ", classsification_accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f_measure)


def test_model(class_Ak_params, test_data):
    """

    :param (class_Ak_param[0]: (For numeric attr): Format of the data is of type
                             {class : {Attribue(Ak-1): (mean, sd)},......., {Attribute(Ak): (mean, sd)}}

            class_Ak_param[1]: (For categorical attr):
                            {class: {Attribute1: {uniqueAttr1: val, uniqueAttr2: value}, {Attribute2: {...}}}}

           class_Ak_param[2] :  The priori probablities for classes in the data, as a dictionary),

           test_data : Absolute or relative path to the file, containing the test data

    :Note  For numerical dataset, Gaussian distribution function is used to obtain parameters for prediction

    :return: prediction of class label for given tuple X
    """

    # The classifier predicts the class label Ci, for a given tuple if and only if

    # P(X | Ci)P(Ci) > P(X | Cj)P(Cj) for 1 ≤ j ≤ m, j != i.

    gaussian_params = class_Ak_params[0]

    categorical_params = class_Ak_params [1]

    collection_PCi = class_Ak_params[2]



    xk_Ci = {}  # For each attributes of tuple X, store the P(xk | Ci)

    # To find the P(X | Ci), we compute P(x1 | Ci), P(x2 |Ci)........,P(xk, Ci)

    # Take the mean , and standard deviation of values of  attributes A1, A2, .....Ak and
    # call gaussioan function
    # Store the result P(X | Ci) as product of each of the results for each class Ci
    # Return the greatest value as the predicted class, Voila.

    pred_for_tuple_x = []  # Contains prediction for each value

    with open(test_data) as test_file:

        lines = test_file.readlines()

        for line in lines:  # For each tuple

            xk = line.split(" ")

            p_xk_ci_ci = {}  # Contains final probability value for i classes of Ci

            if categorical_params:

                for k, v in categorical_params.items():

                    class_conditional_prob = 1.0

                    for ak in range(len(xk) - 1):

                        class_conditional_prob  *= v[ak][xk[ak]]

                    p_xk_ci_ci.update({k: class_conditional_prob * collection_PCi[k]})

            else:

                for k, v in gaussian_params.items():  # For each class Ci

                    class_conditional_prob = 1.0

                    for ak in range(len(xk) - 1):  # For each of the attribute values

                        m, s = v[ak][0], v[ak][1]  # mean and standard deviation

                        class_conditional_prob *= gaussian(float(xk[ak]), m, s)

                    p_xk_ci_ci.update({k: class_conditional_prob * collection_PCi[k]})



            res = (lambda x: max([c_x for c_x in x.values()]))(p_xk_ci_ci)

            for k2, v2 in p_xk_ci_ci.items():
                if v2 == res:
                    pred_for_tuple_x.append(k2)

    # print(pred_for_tuple_x)
    evaluate_performance(test_data, pred_for_tuple_x)


def train_model(train_dataset, datatype):
    """ This method trains a model (or, hypothesis) using Naive Bayes Classification Approach
    :param train_dataset -> Path to the training dataset
            data_type -> Type of dataset, which could be numerical or categorical

            Note: Program assumes that the provided dataset does not include attribute names and,
                  last column in the dataset is always supposed to be a class label

    :return Calculated dict with {class label:{attr: (mean, sd)}} and priori probablities for those class labels """

    class_mean_sd = {}  # Type of data -> {Ci-1 : (attribute name, mean_ci,sd_ci, Ci: (attribute......)  ))}

    class_training_params = {}

    priori_prob_class = {}  # Priori probablity for a given class Ci

    df = pandas.read_table(train_dataset, sep=" ", header=None)

    attributes = df.columns[:-1]  # Get all the attributes of the data set, except for the class

    total_training_tuples = len(df.index)

    predicted_class = df.iloc[:, -1]

    class_labels = predicted_class.unique()  # Class labels C0, C1, C2.........CN , i.e., Ci (general representation)
    # type is numpy.ndarray

    # For each attribute Ak, where k = 1,2....n  and, class label Ci,  we need to compute P (Ak | Ci)

    for i_class in class_labels:

        Ci = df[df.iloc[:, -1] == i_class]

        # print(Ci)

        P_Ci = len(Ci.index) / total_training_tuples

        priori_prob_class.update({i_class: P_Ci})

        if datatype == "numeric":

            # Calculate mean and standard deviation for each attributes with respect to each class
            class_mean_sd.update({i_class: {attr: (Ci[attr].mean(), Ci[attr].std()) for attr in attributes}})

        elif datatype == "categorical":

            class_training_params.update({i_class: {attr: {str(uq_attr): len(Ci.loc[Ci[attr] == uq_attr]) / len(Ci.index)
                                                           for uq_attr in df[attr].unique()}
                                          for attr in attributes}})

    # print(class_training_params)

    return class_mean_sd, class_training_params, priori_prob_class


if __name__ == "__main__":
    # create a parser and parse the command line arguments

    parser = argparse.ArgumentParser(description="Naive Bayessian Classifier")

    parser.add_argument('-m', '--method', type=str, choices=['numeric', 'categorical'],
                        default='numeric',
                        help="Type of data for Naive Bayessian Classifier, numeric or categorical")

    parser.add_argument('-t', '--training', type=str, default="./training.txt", help="Absolute or relative "
                                                                                     "path of training data")

    parser.add_argument('-o', '--testing', type=str, default="./testing.txt", help="Absolute or relative "
                                                                                   "path of test data")

    args = parser.parse_args()

    calculated_probablities = train_model(args.training, args.method)

    prediction = test_model(calculated_probablities, args.testing)
