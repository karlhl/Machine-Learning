# Regression

Regression 就是找到一个函数 function，通过输入特征 x ，输出一个数值 **Scalar** 。

- 股市预测（Stock market forecast）

　　　　输入：过去10年股票的变动、新闻咨询、公司并购咨询等
　　　　输出：预测股市明天的平均值

- 自动驾驶（Self-driving Car）

　　　　输入：无人车上的各个sensor的数据，例如路况、测出的车距等
　　　　输出：方向盘的角度

- 商品推荐（Recommendation）

　　　　输入：商品A的特性，商品B的特性
　　　　输出：购买商品B的可能性

**步骤:**

1. model 模型——线性模型

   ![image-20200827185346639](https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200827185346639.png)

2. goodness of function（确定评价函数）——损失函数

   1. 训练数据

   2. 确定损失函数

      也就是使用 损失函数（Loss function） 来衡量模型的好坏，和越小模型越好。

      ![image-20200827185749719](https://gitee.com/karlhan/picgo/raw/master/img//20200830205814.png)

   3. best function（找出最好的一个函数）——梯度下降法

      找到损失函数最小时候的参数。

      ![image-20200827185843007](https://gitee.com/karlhan/picgo/raw/master/img//20200830205815.png)

      在单变量的函数中，梯度其实就是函数的微分，代表着函数在某个给定点的切线的斜率

      在多变量函数中，梯度是一个向量，向量有方向，梯度的方向就指出了函数在给定点的上升最快的方向

      ![image-20200827190008876](https://gitee.com/karlhan/picgo/raw/master/img//20200830205816.png)

      　　首先在这里引入一个概念 学习率 ：移动的步长，如图7中 η

      　　步骤1：随机选取一个 w0 。
      　　步骤2：计算微分，也就是当前的斜率，根据斜率来判定移动的方向。
      　　　　　大于0向右移动（增加w）
      　　　　　小于0向左移动（减少w）
      　　步骤3：根据学习率移动。
      　　重复步骤2和步骤3，直到找到最低点。

      ![image-20200827190049568](https://gitee.com/karlhan/picgo/raw/master/img//20200830205817.png)	

      两个参数（w,b）

      ![image-20200827190124297](https://gitee.com/karlhan/picgo/raw/master/img//20200830205818.png)

      ​	梯度下降法的问题：只能达到局部最优、

      ![image-20200827190213695](https://gitee.com/karlhan/picgo/raw/master/img//20200830205803.png)	

      欠拟合：指模型拟合程度不高，数据距离拟合曲线较远，或指模型没有很好地捕捉到数据特征，不能够很好地拟合数据。

      过拟合：过拟合是指为了得到一致假设而使假设变得过度严格。

      提高拟合的方法，除了提高模型的复杂度，可以考虑分段函数：

      ![image-20200827190529958](https://gitee.com/karlhan/picgo/raw/master/img//20200830205759.png)

   进一步防止过拟合：**正则化**

   比如先考虑一个参数w，正则化就是在损失函数上加上一个与w（斜率）相关的值，那么要是loss function越小的话，w也会越小，w越小就使function更加平滑（function没那么大跳跃）

   ![image-20200827190747545](https://gitee.com/karlhan/picgo/raw/master/img//20200830205755.png)	

   正则化虽然能够减少过拟合的现象，但是因为加在损失函数后面的值是平白无故加上去的，所以正则化过度的话会导致bias偏差增大

   

参考资料：

​	1. [李宏毅Regression](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/Regression.pdf)











