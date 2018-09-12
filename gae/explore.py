import numpy as np
import scipy.io
from scipy.sparse import csc_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing


def process_attribute(data_source):
    data = scipy.io.loadmat("data/{}.mat".format(data_source))

    labels = data["Label"]
    attributes = data["Attributes"]
    network = data["Network"].toarray()

    pca = PCA(n_components=20)
    new_attributes = pca.fit_transform(attributes)

    data["Label"] = labels
    data["Attributes"] = new_attributes
    data["Network"] = csc_matrix(network)

    scipy.io.savemat("../Data/" + data_source + "_test.mat", data)


def pre_process(data_source):
    data = scipy.io.loadmat("raw_data/{}.mat".format(data_source))
    # labels = data["gnd"]
    # network = data["A"]
    # attributes = data["X"]

    labels = data["Label"]
    attributes = data["Attributes"].toarray()
    network = data["Network"].toarray()

    count = 0
    for label in labels:
        if label[0] == 1:
            count += 1

    min_max_scaler = preprocessing.MinMaxScaler()
    new_attributes = min_max_scaler.fit_transform(attributes)


    newdata = dict()
    newdata["Label"] = labels
    newdata["Network"] = csc_matrix(network)
    newdata["Attributes"] = new_attributes

    scipy.io.savemat("data/{}.mat".format(data_source), newdata)


def explore():
    data = scipy.io.loadmat("../Data/Embedding_BlogCatalog_final.mat")
    labels = data["Label"]

    counter = {}
    for label in labels:
        if label[0] not in counter:
            counter[label[0]] = 1
        else:
            counter[label[0]] += 1

    for key in counter:
        print key, counter[key]


def convert(data_source_list):
    for name in data_source_list:
        data = scipy.io.loadmat("../Data/Deprecated/" + name + ".mat")
        network = data["A"]
        attributes = data["X"]
        labels = data["Y"]
        new_labels = []
        for label in labels:
            index = np.argmax(label) + 1
            new_labels.append([index])

        newdata = dict()
        newdata["Label"] = new_labels
        newdata["Network"] = csc_matrix(network)
        newdata["Attributes"] = csc_matrix(attributes)
        scipy.io.savemat("../Data/" + name + ".mat", newdata)


if __name__ == '__main__':
    data_list = ["Amazon", "Disney", "Enron", "Flickr"]
    pre_process(data_list[3])
    # process_attribute(data_list[3])