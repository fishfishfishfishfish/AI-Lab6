import numpy
import random
import math


# 从文件读取训练数据
def get_train_data(filename):
    """
    :param filename: string, 训练文件名
    :return: list[list[float]], 训练数据[1,x,x,x,...,y]
    """
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        row = line.split(',')
        for i in range(len(row)):
            row[i] = float(row[i])
        row.insert(0, 1)
        # row = numpy.array(row)
        data.append(row)
    return data


# 将训练数据随机分成平均的几份
def split_dataset(data, num):
    """
    :param data: list[list[float]], 原始的训练集
    :param num: int, 要分成的份数
    :return: list[list[list[float]], 分解后的num个数据集数组
    """
    data_list = []
    val_size = len(data) // num
    for k in range(num - 1):
        t_data = []
        for i in range(val_size):
            t_data.append(data.pop(0))
        data_list.append(t_data)
    data_list.append(data)
    return data_list


# 获取训练集和验证集
def get_train_and_val(dataset, n):
    """
    :param dataset: list[list[list[float]], 分割后的数据集
    :param n: int, 第n份作为验证集
    :return: array[float], array[float], 训练集和验证集
    """
    traindata = []
    for i in range(len(dataset)):
        if i != n:
            traindata += dataset[i]
    valdata = dataset[n]  # 剩下的一份作为验证集
    traindata = numpy.array(traindata)
    valdata = numpy.array(valdata)
    return traindata, valdata


def normalize(data):
    mean_each_col = numpy.mean(data, axis=0)
    std_each_col = numpy.std(data, axis=0)

    row, col = data.shape
    data_t = data.copy()
    for i in range(col):
        if std_each_col[i] != 0:
            data_t[:, i] -= mean_each_col[i]
            data_t[:, i] /= std_each_col[i]
    return data_t


def sigmoid(x):
    return 1/(1+math.e**(-x))


def forward(x, w_hidden, w_output):
    """
    :param x: numpy.array, input
    :param w_hidden: numpy.matrix, hidden weight
    :param w_output: numpy.array, output weight
    :return: float, output
    """
    h = w_hidden.dot(x)
    print(h)
    h = sigmoid(h)
    print(h)
    return h.dot(w_output)


def backward(y, o, h, w_output):
    """
    :param y: float, the actual label
    :param o: float, the forward output
    :param h: array, the hidden output
    :param w_output: array, the weights from h to o
    :return: array[Err_hidden], Err_output
    """
    err_output = (y-o)*o*(1-o)
    err_hidden = err_output*h*(1-h)*w_output
    return err_hidden, err_output


def update_w(eta, w_output, w_hidden, err_output, err_hidden, h, x):
    """
    :param eta: float, step length
    :param w_output: array, weights from h to o
    :param w_hidden: matrix, weights from x to h
    :param err_output: float, Err_output
    :param err_hidden: array, Err_hidden
    :param h: array, output of the hidden layer
    :param x: array, input x
    :return: next w_output, w_hidden
    """
    w_output = w_output + eta*err_output*h
    w_hidden = w_hidden + eta*numpy.outer(err_hidden, x)
    return w_output, w_hidden



HiddenNodes = 3
Data = get_train_data("train.csv")
Data = split_dataset(Data, 10)
TrainData, ValData = get_train_and_val(Data, 1)
TrainRow, TrainCol = TrainData.shape
TrainCol -= 1
WHidden = numpy.ones((HiddenNodes, TrainCol))
WOutput = numpy.ones(HiddenNodes)
NorTrainData = normalize(TrainData[:, 0:TrainCol])
print(NorTrainData[0, :])
print(forward(NorTrainData[0, :], WHidden, WOutput))
