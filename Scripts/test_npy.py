from numpy import *  # 使用numpy的属性且不需要在前面加上numpy
import tensorflow as tf
import numpy as np

test = np.load('D:\Codes\ARTangibleAnimation\LKFilters\data\carseq.npy',
               encoding="latin1")  # 加载文件
doc = open('1.txt', 'a')  # 打开一个存储文件，并依次写入
print(test, file=doc)  # 将打印内容写入文件中


#模型文件（.npy）部分内容如下：由一个字典组成，字典中的每一个键对应一层网络模型参数。（包括权重w和偏置b）
a = {'conv1': [array([[1, 2], [3, 4]], dtype=float32), array([5, 6], dtype=float32)], 'conv2': [
    array([[1, 2], [3, 4]], dtype=float32), array([5, 6], dtype=float32)]}

conv1_w = a['conv1'][0]
conv1_b = a['conv1'][1]
conv2_w = a['conv2'][0]
conv2_b = a['conv2'][1]

print(conv1_w)
print(tf.Variable(conv1_w))
print(conv1_b)
print(tf.Variable(conv1_b))
