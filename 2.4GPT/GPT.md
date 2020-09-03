# GPT

Generative Pre-Training (GPT) 

GPT的参数非常大。ELMO有94M的参数，BERT有340M的参数，GPT-2有1542M的参数

GPT是Transformer的Decoder

<img src="https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200828235052752.png" alt="image-20200828235052752" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200828235125608.png" alt="image-20200828235125608" style="zoom: 50%;" />

在前一个输出后，放到输入，然后预测下一个输出。

![image-20200828235708973](https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200828235708973.png)

GPT-2可以在没有训练资料的条件下可以回答问题，翻译，总结。

其中在翻译方面并不是那么好。

<img src="https://raw.githubusercontent.com/karlhl/Picgo/master/image/image-20200829000124094.png" alt="image-20200829000124094" style="zoom:50%;" />

左边的图：

很多词都会attention到第一个词汇。因为如果一个词不知道attention到哪里的时候，就会默认attention到第一个词汇。所以自己做的时候可以加一个特殊的token，代替第一个词。

















