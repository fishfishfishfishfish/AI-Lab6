import numpy
import matplotlib.pyplot as plt
import math


class NeuralNetwork(object):
    def __init__(self, hidden_node, train_size):
        numpy.random.seed(1)
        self.WHidden = numpy.random.random((hidden_node, train_size))
        self.ThetaH = numpy.random.random(hidden_node)
        self.WOutput = numpy.random.random(hidden_node)
        self.ThetaO = numpy.random.random()

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
        data.append(row)
    return data


# 将训练数据分成平均的几份
def split_dataset(data, num):
    """
    :param data: list[list[float]], 原始的训练集
    :param num: int, 要分成的份数
    :return: list[list[list[float]], 分解后的num个数据集数组
    """
    numpy.random.seed(10)
    data_list = []
    val_size = len(data) // num
    for k in range(num - 1):
        t_data = []
        for i in range(val_size):
            t_data.append(data.pop(numpy.random.randint(0, len(data)-1)))
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


def normalize(train_data, val_data, test_data):
    mean_each_col = numpy.mean(train_data, axis=0)
    std_each_col = numpy.std(train_data, axis=0)

    row, col = train_data.shape
    row, colv = val_data.shape
    row, coltx = test_data.shape
    data_t = train_data.copy()
    data_v = val_data.copy()
    data_tx = test_data.copy()
    for ci in range(col):
        if std_each_col[ci] != 0:
            data_t[:, ci] -= mean_each_col[ci]
            data_t[:, ci] /= std_each_col[ci]
    for ci in range(colv):
        if std_each_col[ci] != 0:
            data_v[:, ci] -= mean_each_col[ci]
            data_v[:, ci] /= std_each_col[ci]
    for ci in range(coltx):
        if std_each_col[ci] != 0:
            data_tx[:, ci] -= mean_each_col[ci]
            data_tx[:, ci] /= std_each_col[ci]
    return data_t, data_v, data_tx


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
    """
    :param x_set: matrix, the input x of one batch
    :param y_set: array, the correct label of correlation x
    :param nn: neural network
    :return:
        err_hidden: matrix, the gradient from input to hidden layer
        err_output: array, the gradient from hidden layer to output
        b_err_hidden: array, the gradient for bias from input to hidden layer
        b_err_output: float, the gradient for bias from hidden layer to output
    """
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


def update_batch(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output):
    """
    :param eta: float, step length
    :param nn: NeuralNetwork: WHidden, ThetaH, WOutput, ThetaO
    :param err_output: array, the gradient from hidden layer to output
    :param err_hidden: matrix, the gradient from input to hidden layer
    :param b_err_hidden: array, the gradient for bias from input to hidden layer
    :param b_err_output: float, the gradient for bias from hidden layer to output
    :return: next neural network
    """
    h_size, col = nn.WHidden.shape
    nn.WOutput = nn.WOutput + eta*err_output
    nn.ThetaO = nn.ThetaO + eta*b_err_output*h_size
    nn.WHidden = nn.WHidden + eta*err_hidden
    nn.ThetaH = nn.ThetaH + eta*b_err_hidden*col
    return nn


def update_w(eta, nn, err_output, err_hidden, h, x):
    """
    :param eta: float, step length
    :param nn: NeuralNetwork: WHidden, ThetaH, WOutput, ThetaO
    :param err_output: float, Err_output
    :param err_hidden: array, Err_hidden
    :param h: array, output of the hidden layer
    :param x: array, input x
    :return: next neural network
    """
    nn.WOutput = nn.WOutput + eta*err_output*h
    nn.ThetaO = nn.ThetaO + eta*err_output
    nn.WHidden = nn.WHidden + eta*numpy.outer(err_hidden, x)
    nn.ThetaH = nn.ThetaH + eta*err_hidden
    return nn


def small_try():
    data_set = get_train_data("small-train.csv")
    data_set = numpy.array(data_set)
    row, col = data_set.shape
    col -= 1
    x_set = data_set[:, 0:col]
    y_set = data_set[:, col]
    hidden_nodes = 2
    nn = NeuralNetwork(hidden_nodes, col)
    nn.print_nn()
    print('----------')
    count = 0
    while count < 7000:
        ci = count%2
        nn = train(x_set[ci], y_set[ci], nn, 1)
        nn.print_nn()
        print('------------')
        count += 1


def train(x, y, nn, eta):
    """
    :param x: array, input
    :param y: float, the correct label of x
    :param nn: formal neural network
    :param eta: float, step length
    :return: new neural network
    """
    o, h = forward(x, nn)
    # print('hidden:', h)
    # print('output:', o)
    err_hidden, err_output = backward(y, o, h, nn.WOutput)
    # print('error hidden:', err_hidden)
    # print('error output:', err_output)
    nn = update_w(eta, nn, err_output, err_hidden, h, x)
    return nn


def train_mini_batch(x, y, nn, eta, batch_size):
    """
    :param x: matrix, train set x
    :param y: array, train set y
    :param nn: neural network
    :param eta: float, step length
    :param batch_size: int, the number of feature vector of a batch
    :return: next neural network
    """
    row, col = x.shape
    head = 0
    tail = batch_size
    while tail < row:
        err_hidden, err_output, b_err_hidden, b_err_output = batch(x[head:tail], y[head:tail], nn)
        nn = update_batch(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output)
        head = tail
        tail = head + batch_size
    err_hidden, err_output, b_err_hidden, b_err_output = batch(x[head:row], y[head:row], nn)
    nn = update_batch(eta, nn, err_hidden, err_output, b_err_hidden, b_err_output)
    return nn


def loss(x, y, nn):
    """
    :param x: matrix, input x
    :param y: array, the correct label of correlated x
    :param nn: neural network
    :return: float, the loss
    """
    res = 0
    row, col = x.shape
    for xi in range(row):
        pre, h = forward(x[xi], nn)
        res += ((y[xi]-pre)**2)/2
    return res/row


def val(x, y, nn):
    """
    :param x: matrix, input x
    :param y: array, the correct label of correlated x
    :param nn: neural network
    :return:
    """
    pre = []
    row, col = x.shape
    for xi in range(row):
        tpre, h = forward(x[xi], nn)
        pre.append(tpre)
    pre = numpy.array(pre)
    mul = y*pre
    res = (mul.mean()-pre.mean()*y.mean())/(pre.std()*y.std())
    return res, pre


def test(test_data, nn):
    f = open("15352049_chenxinyu.csv", 'w')
    row, col = test_data.shape
    for xi in range(row):
        o, h = forward(test_data[xi], nn)
        f.write(str(o) + '\n')


# small_try()
HiddenNodes = 5
Eta = 0.000001
BatchSize = 500
Data = get_train_data("train.csv")
TestX = get_train_data("test.csv")
TestX = numpy.array(TestX)
Data = split_dataset(Data, 10)
TrainData, ValData = get_train_and_val(Data, 0)
TrainRow, TrainCol = TrainData.shape
ValRow, ValCol = ValData.shape
TrainCol -= 1
ValCol -= 1
TrainX = TrainData[:, 0:TrainCol]
TrainY = TrainData[:, TrainCol]
ValX = ValData[:, 0:ValCol]
ValY = ValData[:, ValCol]
# TrainX, ValX, TestX = normalize(TrainX, ValX, TestX)
NN = NeuralNetwork(HiddenNodes, TrainCol)
cnt = 0
mse_t = []
mse_v = []
while cnt < 50000:
    It = cnt % TrainRow
    NN = train(TrainX[It], TrainY[It], NN, Eta)
    # NN = train_mini_batch(TrainX, TrainY, NN, Eta, BatchSize)
    if cnt % 10 == 0:
        mse_t.append(loss(TrainX, TrainY, NN))
        mse_v.append(loss(ValX, ValY, NN))
        print(cnt, mse_t[len(mse_t)-1], mse_v[len(mse_v)-1])
    cnt += 1
NN.print_nn()
Corr, Pre = val(ValX, ValY, NN)
print(Corr)
plt.plot(range(len(mse_t)), mse_t, 'b-', range(len(mse_v)), mse_v, 'g-', )
plt.show()
plt.figure()
plt.plot(range(ValRow), Pre, 'b-', range(ValRow), ValY, 'r-')
plt.show()
test(TestX, NN)
