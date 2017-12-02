import numpy
import random
import math


class NeuralNetwork(object):
    def __init__(self, hidden_node, train_size):
        self.WHidden = numpy.ones((hidden_node, train_size))
        self.ThetaH = numpy.ones(hidden_node)
        self.WOutput = numpy.ones(hidden_node)
        self.ThetaO = 1

    def print_nn(self):
        print('w hidden:', self.WHidden)
        print('bias hidden:', self.ThetaH)
        print('w output:', self.WOutput)
        print('bias output:', self.ThetaO)


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


def forward(x, nn):
    """
    :param x: numpy.array, input
    :param nn: NeuralNetwork: WHidden, ThetaH, WOutput, ThetaO
    :return: float, output; array h
    """
    h = nn.WHidden.dot(x) + nn.ThetaH
    h = sigmoid(h)
    o = h.dot(nn.WOutput) + nn.ThetaO
    # o = sigmoid(o)
    return o, h


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


def update_w(eta, nn, err_output, err_hidden, h, x):
    """
    :param eta: float, step length
    :param nn: NeuralNetwork: WHidden, ThetaH, WOutput, ThetaO
    :param err_output: float, Err_output
    :param err_hidden: array, Err_hidden
    :param h: array, output of the hidden layer
    :param x: array, input x
    :return: next w_output, w_hidden
    """
    nn.WOutput = nn.WOutput + eta*err_output*h
    nn.ThetaO = nn.ThetaO + eta*err_output
    nn.WHidden = nn.WHidden + eta*numpy.outer(err_hidden, x)
    nn.ThetaH = nn.ThetaH + eta*err_hidden
    return nn


def small_try():
    x = numpy.array([1, 0, 1])
    y = 1
    eta = 0.9
    nn = NeuralNetwork(2, 3)
    nn.WHidden = numpy.array([[0.2, 0.4, -0.5], [-0.3, 0.1, 0.2]])
    nn.ThetaH = numpy.array([-0.4, 0.2])
    nn.WOutput = numpy.array([-0.3, -0.2])
    nn.ThetaO = 0.1
    o, h = forward(x, nn)
    err_hidden, err_output = backward(y, o, h, nn.WOutput)
    nn = update_w(eta, nn, err_output, err_hidden, h, x)
    nn.print_nn()


def train(x, y, nn, eta):
    o, h = forward(x, nn)
    error = ((o-y)**2)/2
    err_hidden, err_output = backward(y, o, h, nn.WOutput)
    nn = update_w(eta, nn, err_output, err_hidden, h, x)
    return nn, error


# small_try()
HiddenNodes = 10
Eta = 1
Data = get_train_data("train.csv")
Data = split_dataset(Data, 10)
TrainData, ValData = get_train_and_val(Data, 1)
TrainRow, TrainCol = TrainData.shape
TrainCol -= 1
TrainX = normalize(TrainData[:, 0:TrainCol])
TrainY = TrainData[:, TrainCol]
NN = NeuralNetwork(HiddenNodes, TrainCol)
cnt = 0
while cnt < 10000:
    i = cnt % TrainRow
    NN, Error = train(TrainX[i], TrainY[i], NN, Eta)
    print(Error)
    cnt += 1
# print(forward(NorTrainData[0, :], WHidden, WOutput))
