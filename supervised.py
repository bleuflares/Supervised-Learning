import sys
import numpy as np
import operator
import math
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


fold = 10

def cosine_dist(arr1, arr2):
    dot = np.sum(np.dot(np.array(arr1), np.array(arr2)))
    length1 = np.sum(np.dot(np.array(arr1), np.array(arr2)))
    length2 = np.sum(np.dot(np.array(arr1), np.array(arr2)))
    if length1 == 0 or length2 == 0:
        return 99999999
    else:
        return 1 - dot / (math.sqrt(length1) * math.sqrt(length2))

def get_input():
    file = open("dataset/Admission_Predict_Ver1.1.csv", 'r')
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

def decisionTree(tx, ty, vx, vy, height):
    file = open("decisiontree.csv", "w")
    print("Beginning model complexity analysis for NonBoost...")
    file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testxg_score\n")
    for max_depth in range(height):
        classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
        file.write(str(max_depth + 1) + "," + str(cross_val_score(
        classifier, tx, ty, cv = fold).mean()) + ", ")
        classifier.fit(tx, ty)
        file.write(str(classifier.score(tx, ty)) + ", ")
        file.write(str(classifier.score(vx, vy)) + "\n")

#def NeuralNet(tx, ty, vx, vy):

def kNN(x, y, k_max):
    file = open("knn.csv", "w")
    print("Beginning model complexity analysis for KNN...")
    file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
    batch_size = len(x) / fold
    for k_iter in range(k_max):
        accuracy = []
        counts = 0
        for l in range(fold):
            tx = []
            ty = []
            vx = []
            vy = []
            tx = x[:l * batch_size]
            tx.extend(x[(l + 1) * batch_size:])
            ty = y[:l * batch_size]
            ty.extend(y[(l + 1) * batch_size:])
            vx = x[l * batch_size:(l + 1) * batch_size]
            vy = y[l * batch_size:(l + 1) * batch_size]

            for j in range(len(vx)):
                distance = []
                neighbors = {}
                for i in range(len(tx)):
                    distance.append((cosine_dist(tx[i], vx[j]), i))
                distance.sort(key=lambda x : x[0])
                for k in range(k_iter + 1):
                    if ty[distance[k][1]] in neighbors:
                        neighbors[ty[distance[k][1]]] += 1
                    else:
                        neighbors[ty[distance[k][1]]] = 1
                if(vy[j] == max(neighbors.iteritems(), key=operator.itemgetter(1))[0]):
                    counts += 1
            accuracy.append(counts / float(len(vx)))
        result = str(k_iter + 1) + "," + str(sum(accuracy) / float(len(accuracy))) + "\n"
        print(result)
        file.write(result)

if __name__ == "__main__":
    array = get_input()
    x = array[:, :8].tolist()
    y = array[:, 8].tolist()
    tx = array[:450, :8].tolist()
    ty = array[:450, 8].tolist()
    ty = discrete_convert(ty)
    vx = array[450:, :8].tolist()
    vy = array[450:, 8].tolist()
    vy = discrete_convert(vy)
    decisionTree(tx, ty, vx, vy, 10)
    kNN(x, y, 25)
    

"""

def create_mapper(l):
    return {l[n] : n for n in xrange(len(l))}

country = create_mapper(["US", "France", "Spain", "Italy", "Chile", "Germany", "Portugal", "Australia"])
education = create_mapper(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
marriage = create_mapper(["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = create_mapper(["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = create_mapper(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = create_mapper(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = create_mapper(["Female", "Male"])
country = create_mapper(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

converters = {
    1: lambda x: workclass[x] or 8,
    3: lambda x: education[x],
    5: lambda x: x,
    6: lambda x: occupation[x],
    7: lambda x: relationship[x],
    8: lambda x: race[x],
    9: lambda x: sex[x],
    13: lambda x: country[x],
    14: lambda x: income[x]
}


def get_input():
    file = open("/dataset/winemag-data_first150k.csv", 'r')
    data = [line for line in file]

    
    np.loadtxt(data,
                      delimiter=', ',
                      converters=converters,
                      dtype='u4',
                      skiprows=1
                      )
    return tx, ty

"""    