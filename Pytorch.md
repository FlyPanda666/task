# 深度学习框架

## 0. Pytorch

### 1. torch.nn 包

### 2. torch.optim 包

- 优化算法.

> 优化算法的基类是Optimizer,以单独的模块存在;所有的子类也是单独的模块.

- 学习率调整策略.

> 学习率调整的基类是_LRScheduler,基类和其所有的子类放在一个模块中

### 3. torch.nn.modules 包

- module这个模块就是定义Module类的。

[详细分析](https://zhuanlan.zhihu.com/p/88712978)

- 激活函数在该包的activation模块中.

## 1. 梯度清零

在pytorch中，**每个batch训练完之后**需要使用grad_zero_()进行梯度清零。在pytorch中之所以需要手动进行梯度清零，而不是选择自动清零，是因为这种方式可以让使用者自由选择梯度清零的时机，具有更高的灵活性。

> 例如选择训练每N个batch后再进行梯度更新和清零,这相当于将原来的batch_size扩大为N×batch_size.因为原先是每个batch_size训练完后直接更新,而现在变为N个batch_size训练完才更新,相当于将N个batch_size合为了一组.这样可以让使用者使用较低的配置,跑较高的batch_size。

## nn.dropout

作用: 在训练神经网络时,为了解决过拟合问题而随机丢弃一部分元素的方法.

```python
import torch
import torch.nn as nn


layer = nn.Dropout(p=0.3, inplace=False)
inputs = torch.randn(2, 3, 4)
print(inputs)
output = layer(inputs)
print(inputs)
print(output)
```

Dropout函数只有两个参数,第一个是dropout的概率,第二个是参数表示是否在原来的张量上进行修改.
dropout之后,输入的数值发生了变化,每个元素变为 **x/(1-p)**.因为我们要保证输入前后的期望不发生变化.
> 对于数据x, 经过dropout之后的期望为E(x) = x*p + x*(1-p) = x*(1-p).

## 学习率

在pytorch框架中,学习率在torch.optim包的lr_scheduler模块中.

**梯度下降**算法需要我们指定一个学习率作为权重更新步幅的控制因子,学习率越大则权重更新越快.一般来说,我们希望在**训练初期学习率大一些,使得网络收敛迅速,在训练后期学习率小一些,使得网络更好的收敛到最优解**.

学习率衰减类型主要包括:

- 指数衰减
- 固定步长衰减
- 多步长衰减
- 余弦退火衰减

首先要确定的是需要对哪个优化器执行学习率动态调整策略,即先定义一个优化器.

```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```

### 1. 指数衰减

学习率按照指数的形式衰减是比较常用的策略.
对定义好的优化器绑定一个指数衰减学习率控制器.

```python
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
```

> 参数gamma表示衰减的底数,选择不同的gamma值可以获得幅度不同的衰减曲线.

### 2. 固定步长衰减

学习率每隔一定步数（或者epoch）就减少为原来的gamma分之一.

```python
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.65)
```

> 参数gamma参数表示衰减的程度,step_size参数表示每隔多少个step进行一次学习率调整.

### 3. 多步长衰减

不同的区间采用不同的更新频率,或者是有的区间更新学习率,有的区间不更新学习率.

```python
MulStepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 320, 340, 200], gamma=0.8)
```

> 参数milestones为表示学习率更新的起止区间,在区间[0. 200]内学习率不更新，而在[200, 300]、[300, 320].....[340, 400]的右侧值都进行一次更新,gamma参数表示学习率衰减为上次的gamma分之一.

### 4. 余弦退火衰减

余弦退火策略不应该算是学习率衰减策略,因为它使得学习率按照周期变化.

```python
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
```

> 参数T_max表示余弦函数周期,eta_min表示学习率的最小值,默认它是0表示学习率至少为正值.确定一个余弦函数需要知道最值和周期,其中周期就是T_max,最值是初试学习率.

## warmup策略

### 1. 什么是warmup

Warmup是在ResNet论文中提到的一种学习率预热的方法,它在训练开始的时候先选择使用一个较小的学习率,训练了一些epoches或者steps(比如4个epoches,10000steps),**再修改为预先设置的学习率来进行训练**.使用SGD训练神经网络时，在初始使用较大学习率而后期切换为较小学习率是一种广为使用的做法.

### 2. 为什么使用warmup

由于刚开始训练时,模型的权重(weights)是随机初始化的,此时若选择一个较大的学习率,可能带来模型的不稳定(振荡),选择Warmup预热学习率的方式,可以使得开始训练的几个epoches或者一些steps内学习率较小,在预热的小学习率下,模型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快,模型效果更佳。

- 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象,保持分布的平稳.
- 有助于保持模型深层的稳定性.

> Resnet论文中使用一个110层的ResNet在cifar10上训练时,先用**0.01的学习率**训练直到训练误差低于80%(大概训练了400个steps),**然后使用0.1的学习率**进行训练。
> 当我们的mini-batch增大的时候,learning rate也可以成倍增长,即mini-batch大小乘以k，lr也可以乘以k.

### 3. gradual warmup

上例中的Warmup是constant warmup,它的不足之处在于**从一个很小的学习率一下变为比较大的学习率**可能会导致训练误差突然增大.于是18年Facebook提出了gradual warmup来解决这个问题,即从最初的小学习率开始,每个step增大一点点,直到达到最初设置的比较大的学习率时,采用最初设置的学习率进行训练。

```python
"""
Implements gradual warmup, if train_steps < warmup_steps, the
learning rate will be `train_steps/warmup_steps * init_lr`.
Args:
    warmup_steps:warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
    train_steps:训练了的步长数
    init_lr:预设置学习率
"""
import matplotlib.pyplot as plt
warmup_steps = 2500
init_lr = 0.1
# 模拟训练15000步
lr_list = [0.000]
max_steps = 15000
for train_steps in range(max_steps):
    if warmup_steps and train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        warmup_learning_rate = init_lr * warmup_percent_done  # gradual warmup_lr
        learning_rate = warmup_learning_rate
    else:
        # learning_rate = np.sin(learning_rate)  # 预热学习率结束后,学习率呈sin衰减
        learning_rate = learning_rate**1.0001  # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)

    if (train_steps+1) % 100 == 0:
        lr_list.append(learning_rate)
        print("train_steps:%.3f--warmup_steps:%.3f--learning_rate:%.3f" % (
            train_steps+1, warmup_steps, learning_rate))

x = list(range(0, max_steps + 1, 100))
plt.plot(x, lr_list)
plt.show()

```

使用Warmup预热学习率的方式,即先用最初的小学习率训练,然后每个step增大一点点,直到**达到最初设置的比较大的学习率**时(注：此时预热学习率完成),采用最初设置的学习率进行训练(注：预热学习率完成后的训练过程，学习率是衰减的),有助于使模型收敛速度变快，效果更佳。

## torch基础

```python
import torch
from collections import OrderedDict


t = torch.arange(1, 1001, dtype=torch.float)
x = torch.sin(0.01 * t) + torch.normal(0, 0.02, (1000,))


# dl2中定义的函数.
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器,data_arrays可以转入一个元组,一个是特征,另一个是标签.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def init_weights(m):
    if type(m) == nn.modules.linear.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    """定义一个网络,拥有两个全连接层的感知机,ReLU作为激活函数.
    """
    net = nn.modules.container.Sequential(
        OrderedDict([
            ("first", nn.modules.linear.Linear(4, 10)),
            ("active", nn.modules.activation.ReLU()),
            ("second", nn.modules.linear.Linear(10, 1))
            ]
            )
    )
    net.apply(init_weights)
    return net
```

## Linux

### 1. 创建软连接

Linux软链接,类似于windows系统的快捷键。

- 创建软链接

ln -s 【目标目录】 【软链接地址】
> 【目标目录】指软连接指向的目标目录下,【软链接地址】指“快捷键”文件名称,该文件是被指令创建的。如下示例，public文件本来在data文件下是不存在的，执行指令后才存在的。
> ln -s /upload /data/public
注意: 软链接创建需要同级目录下没有同名的目录. 如果存在同名的目录,新创建的软链接会放在该目录下.如果当前目录下存在同名的文件会提示**ln: data: File exists**.

- 删除软链接

rm -rf 【软链接地址】
注意: 上述指令中,软链接地址最后不能含有“/”,当含有“/”时,删除的是软链接目标目录下的资源,而不是软链接本身。

- 修改软链接

ln -snf 【新目标目录】 【软链接地址】
这里修改是指修改软链接的目标目录.
> ln -snf ./Desktop/data home_data
