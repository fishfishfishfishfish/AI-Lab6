import numpy
import matplotlib.pyplot as plt
import math


class NeuralNetwork(object):
    def __init__(self, hidden_node, train_size):
        self.WHidden = numpy.zeros((hidden_node, train_size))
        self.ThetaH = numpy.zeros(hidden_node)
        self.WOutput = numpy.zeros(hidden_node)
        self.ThetaO = 0

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
    err_output = (y-o)
    err_hidden = err_output*h*(1-h)*w_output
    return err_hidden, err_output


def batch(x_set, y_set, nn):
    batch_size, col = x_set.shape
    h_size = len(nn.ThetaH)
    err_hidden = numpy.zeros((h_size, col))
    err_output = numpy.zeros(h_size)
    b_err_hidden = numpy.zeros(h_size)
    b_err_output = 0
    for xi in range(batch_size):
        o, h = forward(x_set[xi], nn)
        terr_hidden, terr_output = backward(y_set[xi], o, h, nn.WOutput)
        err_output += terr_output * h
        err_hidden += numpy.outer(terr_hidden, x_set[xi])
        b_err_output += terr_output
        b_err_hidden += terr_hidden
    return err_hidden, err_output, b_err_hidden, b_err_output


def update_w(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output):
    """
    :param eta: float, step length
    :param nn: NeuralNetwork: WHidden, ThetaH, WOutput, ThetaO
    :param err_output: array, Err_output
    :param err_hidden: array, Err_hidden
    :param h: array, output of the hidden layer
    :param x: array, input x
    :return: next w_output, w_hidden
    """
    h_size, col = nn.WHidden.shape
    nn.WOutput = nn.WOutput + eta*err_output
    nn.ThetaO = nn.ThetaO + eta*b_err_output*h_size
    nn.WHidden = nn.WHidden + eta*err_hidden
    nn.ThetaH = nn.ThetaH + eta*b_err_hidden*col
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


def train(x, y, nn, eta, batch_size):
    row, col = x.shape
    head = 0
    tail = batch_size
    while tail < row:
        err_hidden, err_output, b_err_hidden, b_err_output = batch(x[head:tail], y[head:tail], nn)
        nn = update_w(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output)
        head = tail
        tail = head + batch_size
    err_hidden, err_output, b_err_hidden, b_err_output = batch(x[head:row], y[head:row], nn)
    nn = update_w(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output)
    return nn


def loss(x, y, nn):
    res = 0.0
    for xi in range(len(x)):
        pre, h = forward(x[xi], nn)
        res += ((y[xi]-pre)**2)/2
    return res/len(x)


# small_try()
HiddenNodes = 10
Eta = 0.00000001
BatchSize = 1000
Data = get_train_data("train.csv")
Data = split_dataset(Data, 10)
TrainData, ValData = get_train_and_val(Data, 1)
TrainRow, TrainCol = TrainData.shape
TrainCol -= 1
TrainX = TrainData[:, 0:TrainCol]
TrainY = TrainData[:, TrainCol]
NN = NeuralNetwork(HiddenNodes, TrainCol)
cnt = 0
mse = []
while cnt < 5000:
    i = cnt % TrainRow
    NN = train(TrainX, TrainY, NN, Eta, BatchSize)
    # NN.print_nn()
    mse.append(loss(TrainX, TrainY, NN))
    if cnt % 100 == 0:
        print(cnt)
    cnt += 1
NN.print_nn()
plt.plot(range(len(mse)), mse)
plt.show()
# print(forward(NorTrainData[0, :], WHidden, WOutput))
