# 数据挖掘与知识发现——第五次作业

韩璐-通院-20011210450

**目的：**

* 编程实现K-means算法针对UCI的waveform数据集中每类数据取100个；

  对一副无噪图像进行分割

*  编程实现PAM对部分waveform数据集加20%的高斯噪声；

  同时对一副噪声图像进行分割

**数据集**

Waveform Database Generator (Version 2) Data Set

### 一、实验原理

聚类算法是指将一堆没有标签的数据自动划分成几类的方法。

**K-Means**

算法的特点是类别的个数是人为给定的，K-Means的一个重要的假设是：数据之间的相似度可以使用欧氏距离度量。

步骤：

（1）随机选取k个质心（k值取决于你想聚成几类）
（2）计算样本到质心的距离，距离质心距离近的归为一类，分为k类
（3）求出分类后的每类的新质心
（4）再次计算计算样本到新质心的距离，距离质心距离近的归为一类
（5）判断新旧聚类是否相同，如果相同就代表已经聚类成功，如果没有就循环2-4步骤直到相同

**PAM**

围绕中心点划分，Partitioning Around Medoids，首先为每个簇随意选择选择一个代表对象, 剩余的对象根据其与代表对象的距离分配给最近的一个簇; 然后反复地用非代表对象来替代代表对象，以改进聚类的质量 。

步骤：

  (1)  随机选择k个对象作为初始的代表对象；
  (2)  repeat
	  (3)  指派每个剩余的对象给离它最近的代表对象所代表的簇；
	  (4)  随意地选择一个非代表对象$O_{random}$；
	  (5)  计算用$O_{random}$代替$O_{j}$的总距离E, 如果E比取代前下降则则用$O_{random}$替换$O_{j}$，形成新的k个代表对象的集合，返回（4）；
      (6) until 不发生变化
  (7) 如果所有非代表对象都无法取代已存在的簇中心，则结束替代过程，并输出结果

### 二、实验方案

##### 1.1 实现K-means

waveform数据集有5000样本，有21个特征，作者将其分为三类，并进行了标明，用pd.csv()函数读取结果如下：

###### 1）自己设计实现

加载数据

```python
def loadDataSet(fileName):
    data= pd.read_csv(fileName,header=None)
    print(data)
    return data
```

欧式距离

```python
def distEclud(arrA,arrB):
    d=arrA-arrB;
    return np.sum(np.power(d,2),axis=1)
```

随机初始化中心

```python
def randCent(dataSet,k):
    n = dataSet.shape[1]
    data_min = dataSet.iloc[:,:n-1].min()
    data_max = dataSet.iloc[:,:n-1].max()
    # 输出维度(k,n-1)
    centre = np.random.uniform(data_min,data_max,(k,n-1))
    return centre
```

KMeans实现

```python
def KMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    """
    :param dataSet: 数据集路径
    :param k: 选择的分类值数量
    :param distMeas: 距离选取，这里是欧式距离
    :param createCent: 采用随机初始化
    :return:返回中心，
    """
    m,n= dataSet.shape  #行的数目
    # 随机三个中心值，每个是n-1维的
    centroids = createCent(dataSet,k)
    # 第一列存样本的到簇的中心点的误差的最小值
    # 第二列存上一次样本属于哪一簇
    # 第三列存最新一次样本属于那一簇
    clusterAssment = np.zeros((m,3))
    clusterAssment[:,0] = np.inf
    clusterAssment[:,1:3] = -1
    result_set=pd.concat([dataSet,pd.DataFrame(clusterAssment)],axis=1,ignore_index=True)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有的样本（行数）
        for i in range(m):
            # 遍历所有的质心
            #第2步 找出最近的质心
            # 计算该样本到质心的欧式距离
            distance = distMeas(dataSet.iloc[i,:n-1].values,centroids)
            # 第 3 步：更新每一行样本所属的簇
            result_set.iloc[i,n] = distance.min()
            result_set.iloc[i,n+1] = np.where(distance==distance.min())[0]
        clusterChanged = not(result_set.iloc[:,-1] ==result_set.iloc[:,-2]).all()
        #第 4 步：更新质心
        if clusterChanged:
            cent_df= result_set.groupby(n+1).mean()
            centriods = cent_df.iloc[:,:n-1].values
            result_set.iloc[:,-1]=result_set.iloc[:,-2]
    print("Congratulations,cluster complete!")
    return centroids,result_set
```

首先初始化中心，在结尾设置了三列。第一列是该行到中心点的最小值，第二列是上一次样本属于哪一个分类，第三列是最新一次属于哪个分类。然后开始循环迭代，不停寻找最佳的中心。循环为止条件是如果上一次和最新一次的分类都相同，就可以认为分类已经收敛，不会再更新了，所以认为完成分类了。

结果：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023152925078.png" alt="image-20201023152925078"  />

跟真实值进行对比，还是有出入的。

###### 2）使用Sklearn实现

```python
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("waveform.data",header=None)
print(data)

n = data.shape[1] # 一共22列

train_x = data.iloc[:,0:n-1]
# print(train_x)

kmeans = KMeans(n_clusters=3)#n_clusters=3即指定划分为3个类型
kmeans.fit(train_x)#模型训练
y_kmeans = kmeans.predict(train_x)#模型预测
print(y_kmeans)
```

直接套用即可

![image-20201023155000817](https://gitee.com/karlhan/picgo/raw/master/img//image-20201023155000817.png)

经过比对，发现效果也不是很好，可能是只进行比对了开头几个。

##### 1.2 对无噪图进行分割

使用了sklearn进行分割

```python
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing

# 加载图像，并对数据进行规范化
def load_data(filePath):
    # 读文件
    f = open(filePath,'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点(x,y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height

# 加载图像，得到规范化的结果img，以及图像尺寸
img, width, height = load_data('无噪.jpg')
```

读取文件函数，然后计算出像素值，以及长宽。对每个像素值进行遍历，得到彩色的三个通道值，并进行归一化。

进行聚类

```python
# 用K-Means对图像进行3聚类
if __name__ == "__main__":
    kmeans =KMeans(n_clusters=3)
    kmeans.fit(img)
    label = kmeans.predict(img)
    # 将图像聚类结果，转化成图像尺寸的矩阵
    label = label.reshape([width, height])
    # 创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
    pic_mark = image.new("L", (width, height))
    for x in range(width):
        for y in range(height):
            # 根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
            pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
    pic_mark.save("3_mark.jpg", "JPEG")
```

就是生成每个像素的label（分类值）值后，生成一个新图像，灰度值是0-255，根据分的类别个数不一样，就是255除分类的个数，实现等分。生成新图像保存。

原图

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023165930000.png" alt="image-20201023165930000" style="zoom: 67%;" />

分为2类

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023170000965.png" alt="image-20201023170000965" style="zoom:67%;" />

分为3类

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023170014463.png" alt="image-20201023170014463" style="zoom:67%;" />

分为4类

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023170026693.png" alt="image-20201023170026693" style="zoom:67%;" />

可以看到细节一步步变多。分到4类时候阴影都可以看见了。

##### 2.1 PAM

对waveform进行加噪和聚类

读取数据以及加噪

产生一个与数据集一样形状的噪声，直接在在数据集上，然后恢复label。

```python
df = np.loadtxt('waveform.data', delimiter=',')  # 载入waveform数据集，22列，最后一列为标签0，1，2
s = np.array(df)
# 产生噪声
noise = 0.2* np.random.randn(s.shape[0],s.shape[1])
noise = noise - np.mean(noise)

# print(noise)

data = df + noise
data[:,-1] = s[:,-1]
print(data.shape)
```

主函数

```python
def dis(data_a, data_b):
    return np.sqrt(np.sum(np.square(data_a - data_b), axis=1))  # 返回欧氏距离

def kmeans_wave(n=10, k=3, data=data):
    data_new = copy.deepcopy(data)  # 前21列存放数据，不可变。最后1列即第22列存放标签，标签列随着每次迭代而更新。
    data_now = copy.deepcopy(data)  # data_now用于存放中间过程的数据

    center_point = np.random.choice(300, 3, replace=False)
    center = data_new[center_point, :20]  # 随机形成的3个中心，维度为（3，21）

    distance = [[] for i in range(k)]
    distance_now = [[] for i in range(k)]  # distance_now用于存放中间过程的距离
    lost = np.ones([300, k]) * float('inf')  # 初始lost为维度为（300，3）的无穷大

    for j in range(k):  # 首先完成第一次划分，即第一次根据距离划分所有点到三个类别中
        distance[j] = np.sqrt(np.sum(np.square(data_new[:, :20] - np.array(center[j])), axis=1))
    data_new[:, 21] = np.argmin(np.array(distance), axis=0)  # data_new 的最后一列，即标签列随之改变，变为距离某中心点最近的标签，例如与第0个中心点最近，则为0

    for i in range(n):  # 假设迭代n次
        for m in range(k):  # 每一次都要分别替换k=3个中心点，所以循环k次。这层循环结束即算出利用所有点分别替代3个中心点后产生的900个lost值
            for l in range(300):  # 替换某个中心点时都要利用全部点进行替换，所以循环300次。这层循环结束即算出利用所有点分别替换1个中心点后产生的300个lost值
                center_now = copy.deepcopy(center)  # center_now用于存放中间过程的中心点
                center_now[m] = data_now[l, :20]  # 用第l个点替换第m个中心点
                for j in range(k):  # 计算暂时替换1个中心点后的距离值
                    distance_now[j] = np.sqrt(np.sum(np.square(data_now[:, :20] - np.array(center_now[j])), axis=1))
                data_now[:, 21] = np.argmin(np.array(distance),
                                            axis=0)  # data_now的标签列更新，注意data_now时中间过程，所以这里不能选择更新data_new的标签列

                lost[l, m] = (dis(data_now[:, :20], center_now[data_now[:, 21].astype(int)]) \
                              - dis(data_now[:, :20], center[data_new[:, 21].astype(
                            int)])).sum()  # 这里很好理解lost的维度为什么为300*3了。lost[l,m]的值代表用第l个点替换第m个中心点的损失值

        if np.min(lost) < 0:  # lost意味替换代价，选择代价最小的来完成替换
            index = np.where(np.min(lost) == lost)  # 即找到min(lost)对应的替换组合
            index_l = index[0][0]  # index_l指将要替代某个中心点的候选点
            index_m = index[1][0]  # index_m指将要被替代的某个中心点，即用index_l来替代index_m

        center[index_m] = data_now[index_l, :20]  # 更新聚类中心

        for j in range(k):
            distance[j] = np.sqrt(np.sum(np.square(data_now[:, :20] - np.array(center[j])), axis=1))
        data_new[:, 21] = np.argmin(np.array(distance), axis=0)  # 更新参考矩阵,至此data_new的标签列得以更新，即完成了一次迭代

    return data_new  # 最后返回data_new，其最后一列即为最终聚好的标签


if __name__ == '__main__':
    # 迭代10次，分为3个label
    data_new = kmeans_wave(10, 3, data)
    print(data_new.shape)
   
```

结果

最后一列是生成的新label

![image-20201023205755836](https://gitee.com/karlhan/picgo/raw/master/img//image-20201023205755836.png)

输出一下原来的标签和新生成的标签，因为新生成的label的序号跟原来的不一定是一样的，所以要比哪几个是一起的来区分结果的好坏。

![image-20201023210535195](https://gitee.com/karlhan/picgo/raw/master/img//image-20201023210535195.png)

##### 2.2 PAM对有噪图分割

原图：

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023212718239.png" alt="image-20201023212718239" style="zoom:67%;" />

分为两类

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023212504128.png" alt="image-20201023212504128" style="zoom:67%;" />

分为3类

<img src="https://gitee.com/karlhan/picgo/raw/master/img//image-20201023212521977.png" alt="image-20201023212521977" style="zoom:67%;" />

跟无噪图对比，图片的墙上多了斑点。

### 总结感悟

本次实验通过手写kmeans和PAM代码，深入理解了聚类的这两种的算法原理，加深了理解。并对图片进行了分割的应用。



