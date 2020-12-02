# SSDL
## Introduction
This is a reproduction of *A deep learnign framework for hyperspectral image classification using spatial pyramid pooling*.

![image](image/ssdl.jpg)
## Requirements
* pytorch 1.3
* scikit-learn
* scipy
* visdom
## Experiments
模型分别在PaviaU，Salinas和KSC这三个基准数据集上进行测试。实验总共分为三组，分别为每类样本量为10，每类样本量为50和每类样本量为100。为了减少误差，每组实验分别进行10次，最终的准确率取10次实验的均值。

在PaviaU数据集上的准确率（%）如下表所示：
<table>
<tr align="center">
<td colspan="6">PaviaU</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>74.79</td>
<td>2.34</td>
<td>84.92</td>
<td>1.34</td>
<td>89.33</td>
<td>1.36</td>
</tr>
</table>

在Salinas数据集上的准确率（%）如下表所示：
<table>
<tr align="center">
<td colspan="6">Salinas</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>74.29</td>
<td>4.55</td>
<td>85.79</td>
<td>1.91</td>
<td>88.67</td>
<td>1.13</td>
</tr>
</table>

在KSC数据集上的准确率（%）如下表所示：
<table>
<tr align="center">
<td colspan="6">KSC</td>
</tr>
<tr align="center">
<td colspan="2">10</td>
<td colspan="2">50</td>
<td colspan="2">100</td>
</tr>
<tr align="center">
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
<td>mean</td>
<td>std</td>
</tr>
<tr align="center">
<td>83.71</td>
<td>2.76</td>
<td>96.88</td>
<td>0.84</td>
<td>98.75</td>
<td>0.29</td>
</tr>
</table>

## Runing the code
划分数据集 `python trainTestSplit.py`

训练模型 `python CrossTrain.py --name xx --epoch xx --lr xx`
