import sys
import numpy as np
import operator
import math
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


fold = 10

def euc_dist(arr1, arr2):
    diff = np.array(arr1) - np.array(arr2)
    return np.sum(np.dot(diff, diff))

def cos_dist(arr1, arr2):
    length1 = float(np.sum(np.dot(np.array(arr1), np.array(arr1))))
    length2 = float(np.sum(np.dot(np.array(arr2), np.array(arr2))))
    return np.sum(np.dot(np.array(arr1), np.array(arr2))) / (length1 * length2)
    

def get_admission_input():
    file = open("dataset/Admission_Predict_Ver1.1.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

def get_accident_input():
    file = open("dataset/Accident2.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

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

def decisionTree(tx, ty, vx, vy, height, data):
    print("DT")
    file = open(data + "decisiontree.csv", "w")
    file.write("max_depth" + ", " + "cross_val_score" + ", " + "train_score" + ", " + "test_score\n")
    for max_depth in range(height):
        classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
        file.write(str(max_depth + 1) + "," + str(cross_val_score(
        classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        file.write(str(classifier.score(tx, ty)) + ", ")
        file.write(str(classifier.score(vx, vy)) + "\n")

def AdaBoosting_depth(tx, ty, vx, vy, n, height, data):
    print("boosting with depth")
    file = open(data + "ada_boosting_max_depth.csv", "w")
    file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for max_depth in range(height):
        classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=100)
        result = ""
        result += (str(max_depth + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)
        print("write")

def AdaBoosting(tx, ty, vx, vy, n, height, data):
    print("boosting with estimator")
    file = open(data + "ada_boosting_estimator_num.csv", "w")
    file.write("max_depth" + ", "  + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for estimator_num in range(n):
        classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=height), n_estimators=(estimator_num + 1))
        result = ""
        result += (str(estimator_num + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)

    print("boosting with depth")
    file = open(data + "ada_boosting_max_depth.csv", "w")
    file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for max_depth in range(height):
        classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=100)
        result = ""
        result += (str(max_depth + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)
        print("write")

def NeuralNet_depth(tx, ty, vx, vy, neurons_num, max_depth, data):
    print("NeuralNet depth")
    scaler = StandardScaler()
    scaler.fit(tx)
    tx = scaler.transform(tx)
    vx = scaler.transform(vx)
    file = open(data + "neural_network_depth.csv", "w")
    file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    layers = []
    for depth in range(max_depth):
        for neuron in range(depth):
            layers.append(neurons_num)
        classifier = MLPClassifier(solver='adam', alpha=1e-5, max_iter=10000, hidden_layer_sizes=(layers), random_state=1)
        result = ""
        result += (str(depth + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)

def NeuralNet_neuron_num(tx, ty, vx, vy, neurons_num, data):
    print(data + "NeuralNet neuron num")
    scaler = StandardScaler()
    scaler.fit(tx)
    tx = scaler.transform(tx)
    vx = scaler.transform(vx)
    file = open(data + "neural_network_neuron_num.csv", "w")
    file.write("neurons_per_layer" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for neuron in range(neurons_num):
        layers = []
        layers.append(neuron + 1)
        classifier = MLPClassifier(solver='adam', alpha=1e-5, max_iter=10000, hidden_layer_sizes=(layers), random_state=1)
        result = ""
        result += (str(neuron + 1) + "," + str(cross_val_score(classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)

def kNN(tx, ty, vx, vy, k_max, data):
    print("kNN")
    file = open(data + "knn.csv", "w")
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

def kNN_fast(tx, ty, vx, vy, k_max, data):
    print("kNN")
    file = open(data + "knn.csv", "w")
    file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    for k in range(k_max):
        classifier = KNeighborsClassifier(n_neighbors = k + 1)
        result = ""
        result += (str(k + 1) + "," + str(cross_val_score(
        classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        result += str(classifier.score(tx, ty)) + ", "
        result += str(classifier.score(vx, vy)) + "\n"
        file.write(result)

def SVM(data):
    print("SVM")
    file = open(data + "svm.csv", "w")
    LinearSVC = svm.LinearSVC();
    RBFSVC = svm.SVC(kernel="rbf")
    SigmoidSVC = svm.SVC(kernel="sigmoid")
    
    result = ""
    result += ("Linear_SVC" + "," + str(cross_val_score(
    LinearSVC, tx, ty, cv = fold).mean()) + ", ")
    LinearSVC.fit(tx, ty)
    result += str(LinearSVC.score(tx, ty)) + ", "
    result += str(LinearSVC.score(vx, vy)) + "\n"
    file.write(result)

    result = ""
    result += ("RBF_SVC" + "," + str(cross_val_score(
    RBFSVC, tx, ty, cv = fold).mean()) + ", ")
    RBFSVC.fit(tx, ty)
    result += str(RBFSVC.score(tx, ty)) + ", "
    result += str(RBFSVC.score(vx, vy)) + "\n"
    file.write(result)

    result = ""
    result += ("Sigmoid_SVC" + "," + str(cross_val_score(
    SigmoidSVC, tx, ty, cv = fold).mean()) + ", ")
    SigmoidSVC.fit(tx, ty)
    result += str(SigmoidSVC.score(tx, ty)) + ", "
    result += str(SigmoidSVC.score(vx, vy)) + "\n"
    file.write(result)

if __name__ == "__main__":
    #print("small input")
    #array = get_admission_input()
    #x = array[:, :8]
    #x = (x / x.max(axis=0)).tolist()
    #y = array[:, 8].tolist()
    #y = discrete_convert(y)
    #tx = x[:450]
    #ty = y[:450]
    #vx = x[450:]
    #vy = y[450:]
    #decisionTree(tx, ty, vx, vy, 25, "admission")
    #AdaBoosting(tx, ty, vx, vy, 10, 25, "admission")
    #kNN(tx, ty, vx, vy, 25, "admission")
    #SVM("admission")
    #NeuralNet_neuron_num(tx, ty, vx, vy, 10, 5, "admission")
    
    print("big input")
    array = get_accident_input()
    tx = array[:len(array) * 9 / 10, :13].tolist()
    ty = array[:len(array) * 9 / 10, 13].tolist()
    vx = array[len(array) * 9 / 10:, :13].tolist()
    vy = array[len(array) * 9 / 10:, 13].tolist()
    #decisionTree(tx, ty, vx, vy, 100, "accident2")
    #kNN_fast(tx, ty, vx, vy, 100, "accident2")
    #SVM("accident2")
    #AdaBoosting_depth(tx, ty, vx, vy, 80, 100, "accident")
    #NeuralNet_depth(tx, ty, vx, vy, 50, 50, "accident2")
    NeuralNet_neuron_num(tx, ty, vx, vy, 100, "accident2")
    
    
