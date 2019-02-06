import sys
import numpy as np
import operator
import math
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


fold = 10

def euc_dist(arr1, arr2):
    diff = np.array(arr1) - np.array(arr2)
    return np.sum(np.dot(diff, diff))

def cos_dist(arr1, arr2):
    length1 = float(np.sum(np.dot(np.array(arr1), np.array(arr1))))
    length2 = float(np.sum(np.dot(np.array(arr2), np.array(arr2))))
    return np.sum(np.dot(np.array(arr1), np.array(arr2))) / (length1 * length2)
    

def get_input():
    file = open("dataset/Admission_Predict_Ver1.1.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

def get_input2():
    file = open("dataset/Accident.csv", 'r')
    return np.loadtxt(file, delimiter=" ", skiprows=1)

def discrete_convert(l):
    n = []
    for x in l:
        if x < 0.25:
            n.append(1)
        elif x < 0.5:
            n.append(2)
        elif x < 0.75:
            n.append(3)
        else:
            n.append(4)
    return n

def decisionTree(tx, ty, vx, vy, height):
    print("DT")
    file = open("decisiontree.csv", "w")
    file.write("max_depth" + ", " + "cross_val_score" + ", " + "train_score" + ", " + "test_score\n")
    for max_depth in range(height):
        classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
        file.write(str(max_depth + 1) + "," + str(cross_val_score(
        classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        file.write(str(classifier.score(tx, ty)) + ", ")
        file.write(str(classifier.score(vx, vy)) + "\n")

def boosting(tx, ty, vx, vy, n, height):
    print("boosting")
    file = open("boosting.csv", "w")
    file.write("max_depth" + ", " + "n_estimators" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for n_estimators in range(n):
        for max_depth in range(height):
            classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=10* (n_estimators + 1))
            result = ""
            result += (str(max_depth + 1) + "," + str(n_estimators + 1) + "," + str(cross_val_score(
            classifier, tx, ty, cv = fold).mean()) + ", ")
            classifier.fit(tx, ty)
            result += str(classifier.score(tx, ty)) + ", "
            result += str(classifier.score(vx, vy)) + "\n"
            print(result)
            file.write(result)

def NeuralNet(tx, ty, vx, vy, neurons_num, max_depth):
    scaler = StandardScaler()
    scaler.fit(tx)
    tx = scaler.transform(tx)
    vx = scaler.transform(vx)
    file = open("aviation_accidents_neural_network_layer_results.csv", "w")
    print("Beginning model complexity analysis for NeuralNetwork... neurons")
    file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    layers = []
    for depth in range(max_depth):
        for neuron in range(depth):
            layers.append(neurons_num)
        classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
        result = ""
        result += (str(depth + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        print(result)
        file.write(result)

def kNN(tx, ty, vx, vy, k_max):
    print("kNN")
    file = open("knn.csv", "w")
    print("Beginning model complexity analysis for KNN...")
    file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    batch_size = len(tx) / fold
    for k_iter in range(k_max):
        accuracy = []
        for l in range(fold):
            counts = 0
            cvtx = []
            cvty = []
            cvvx = []
            cvvy = []
            cvtx = tx[:l * batch_size]
            cvtx.extend(tx[(l + 1) * batch_size:])
            cvty = ty[:l * batch_size]
            cvty.extend(ty[(l + 1) * batch_size:])
            cvvx = tx[l * batch_size:(l + 1) * batch_size]
            cvvy = ty[l * batch_size:(l + 1) * batch_size]
            for j in range(len(cvvx)):
                distance = []
                neighbors = {}
                for i in range(len(cvtx)):
                    distance.append((euc_dist(cvtx[i], cvvx[j]), i))
                distance.sort(key=lambda x : x[0])
                for k in range(k_iter + 1):
                    if cvty[distance[k][1]] in neighbors:
                        neighbors[cvty[distance[k][1]]] += 1
                    else:
                        neighbors[cvty[distance[k][1]]] = 1
                if(cvvy[j] == max(neighbors.iteritems(), key=operator.itemgetter(1))[0]):
                    counts += 1
            accuracy.append(counts / float(len(cvvx)))
        result = str(k_iter + 1) + "," + str(sum(accuracy) / float(len(accuracy)))
        file.write(result)

        counts = 0
        for j in range(len(tx)):
            distance = []
            neighbors = {}
            for i in range(len(tx)):
                distance.append((euc_dist(tx[i], tx[j]), i))
            distance.sort(key=lambda x : x[0])
            for k in range(k_iter + 1):
                if ty[distance[k][1]] in neighbors:
                    neighbors[ty[distance[k][1]]] += 1
                else:
                    neighbors[ty[distance[k][1]]] = 1
            if(ty[j] == max(neighbors.iteritems(), key=operator.itemgetter(1))[0]):
                counts += 1
        result = str(k_iter + 1) + "," + str(counts / float(len(tx)))
        file.write(result)

        counts = 0
        for j in range(len(vx)):
            distance = []
            neighbors = {}
            for i in range(len(tx)):
                distance.append((euc_dist(tx[i], vx[j]), i))
            distance.sort(key=lambda x : x[0])
            for k in range(k_iter + 1):
                if ty[distance[k][1]] in neighbors:
                    neighbors[ty[distance[k][1]]] += 1
                else:
                    neighbors[ty[distance[k][1]]] = 1
            if(vy[j] == max(neighbors.iteritems(), key=operator.itemgetter(1))[0]):
                counts += 1
        result = str(k_iter + 1) + "," + str(counts / float(len(vx))) + "\n"
        file.write(result)

def SVM():
    print("SVM")
    file = open("svm.csv", "w")
    LinearSVC = svm.LinearSVC();
    RBFSVC = svm.SVC(kernel="rbf")
    SigmoidSVC = svm.SVC(kernel="sigmoid")
    
    result = ""
    result += ("Linear_SVC" + "," + str(cross_val_score(
    LinearSVC, tx, ty, cv = fold).mean()) + ", ")
    LinearSVC.fit(tx, ty)
    result += str(LinearSVC.score(tx, ty)) + ", "
    result += str(LinearSVC.score(vx, vy)) + "\n"
    print(result)
    file.write(result)

    result = ""
    result += ("RBF_SVC" + "," + str(cross_val_score(
    RBFSVC, tx, ty, cv = fold).mean()) + ", ")
    RBFSVC.fit(tx, ty)
    result += str(RBFSVC.score(tx, ty)) + ", "
    result += str(RBFSVC.score(vx, vy)) + "\n"
    print(result)
    file.write(result)

    result = ""
    result += ("Sigmoid_SVC" + "," + str(cross_val_score(
    SigmoidSVC, tx, ty, cv = fold).mean()) + ", ")
    SigmoidSVC.fit(tx, ty)
    result += str(SigmoidSVC.score(tx, ty)) + ", "
    result += str(SigmoidSVC.score(vx, vy)) + "\n"
    print(result)
    file.write(result)

if __name__ == "__main__":
    """
    array = get_input()
    x = array[:, :8]
    x = (x / x.max(axis=0)).tolist()
    y = array[:, 8].tolist()
    y = discrete_convert(y)
    tx = x[:450]
    ty = y[:450]
    vx = x[450:]
    vy = y[450:]
    #decisionTree(tx, ty, vx, vy, 10)
    #boosting(tx, ty, vx, vy, 10, 10)
    #kNN(tx, ty, vx, vy, 25)
    SVM()
    NeuralNet(tx, ty, vx, vy, 10, 10)
    """
    array = get_input2()
    #x = array[:, :12].tolist()
    #y = array[:, 12].tolist()
    tx = array[:1000000, 13].tolist()
    ty = array[:1000000, :12].tolist()
    vx = array[1000000:, 13].tolist()
    vy = array[1000000:, :12].tolist()
    print(len(tx))
    print(len(ty))
    print(len(vx))
    print(len(vy))
    decisionTree(tx, ty, vx, vy, 100)
    boosting(tx, ty, vx, vy, 10, 10)
    #kNN(x, y, 25)
    """
