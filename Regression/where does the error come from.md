# Where does the error come form?

误差的定义：理论最佳函数f1 和我们用模型预测出的函数f2 之间的差值

误差的两大来源：

**bias：**估计值的期望等于假设值（如上文的E(m)=u），即为无偏差，反之有偏差（bias）。

当样本数越来越大时，m就越靠近u。

**variance：**方差表示的数据的离散程度，对于方差的估计是有差估计

![image-20200827193223247](https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200827193223247.png)

![image-20200827193242409](https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200827193242409.png)

​	简单的model可能会造成较大的bias,较小的Variance

​	复杂的model会造成较小的bias,较大的Variance，因为训练出来的函数太扭曲了

​	我们理想中的目标是找到一个平衡点，使bias和variance尽可能小。

![image-20200827194940895](C:\Users\Karl\AppData\Roaming\Typora\typora-user-images\image-20200827194940895.png)

​	如何判断是欠拟合还是过拟合？

![image-20200827195120405](C:\Users\Karl\AppData\Roaming\Typora\typora-user-images\image-20200827195120405.png)

​	欠拟合，bias大：需要重新设计model

​	过拟合，方差大：需要增加数据data。或者加入正则化

​	![image-20200827200045154](C:\Users\Karl\AppData\Roaming\Typora\typora-user-images\image-20200827200045154.png)

​	**如何选择模型：**可以先将一个模型分成三份，每次选一份作为测试集，然后进行测试错误率，然后将三个

​	模型的错误率计算平均错误率，选出最佳模型，然后在完整的模型上再测试。

![image-20200827233031934](C:\Users\Karl\AppData\Roaming\Typora\typora-user-images\image-20200827233031934.png)



















