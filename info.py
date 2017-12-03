import numpy
import math


# def normalize(data):
#     mean_each_col = numpy.mean(data, axis=0)
#     std_each_col = numpy.std(data, axis=0)
#
#     row, col = data.shape
#     data_t = data.copy()
#     print(data_t[2][3])
#     print(data_t[3][2])
#     for i in range(col):
#         data_t[:, i] -= mean_each_col[i]
#         data_t[:, i] /= std_each_col[i]
#     return data_t
#
#
a = numpy.array([[1, 2], [3, 4]])
# b = numpy.array([1, 2, 3])
# print(numpy.mean(a, axis=0))
# print(numpy.mean(b))
# a = numpy.array([[1.0, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9], [7, 8, 9, 10, 11]])
# print(normalize(a))


# a = numpy.array([1, 2, 3, 4])
b = numpy.array([1, 2, 3])
a += a
print(a)
c = numpy.array([2, 3, 4, 5])
# print(a*(1-a)*c)

