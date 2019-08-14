## $$\color{lime}{An\ paper\ a\ day\ keeps\ the\ trouble\ away\ !}$$

学习生成对抗网络，第一件事就是看这篇神作《Generative Adversarial Nets》，下面对这篇论文做一个学习的总结，主要关注于文中介绍生成对抗网络的部分，其余内容详见论文，它对于我们理解生成对抗网络不会有太大的影响。 <center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190329215042848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
___
## 总览
___

看一篇论文最重要的自然是先看摘要部分，下面我们看一下它对本文的主要内容是如何介绍的。

在本文中作者提出了一种新的框架，通过对抗的过程来评估生成模型，在这个过程中同时训练以下两个模型:
- 生成器（generative model，G）：捕捉样本的分布规律
- 判别器（discriminative model ，D）：判断输入的样本事来自真实的训练数据还是生成器G

G 的训练过程是让D犯错的概率最大化，作者提出的这个新的框架相当于一个极小化极大的双方博弈游戏的过程。在任意的G和D的空间中存在唯一的解，使得G可以复现训练数据分布，并且D判断的概率处处都等于$\frac {1}{2}$，即使得D难以分辨数据是真实的还是G生成的。

当G和D都由多层感知机来定义时，我们就可以通过利用反向传播来进行训练。同时在训练和生成样本的过程中，相比于之前的相关算法，它不需要任何的马尔科夫链和近似的推理网络。

在$Introduction$和$Related work$作者介绍了和本文内容相关的一些知识，包括之前广泛使用的其他的相关的算法，具体可见论文相应部分。

___
## 对抗网络
___

下面我们根据对抗网络的思想，从数学公式的推导上对其做一个理性的认识，来逐步的推出文中的这个公式$$\min \limits_{G} \max \limits _{D} V(G,D)=E_{x \sim p_{data}(x) } logD(x_{i})+E_{x \sim p_{z}(z)}log(1-D(G(z_{i})))$$


假设用$x_{i}$来表示真实的数据样本，$z_{i}$表示噪声数据，其中$i=1,2,…,m$，$p_{x}(x)$表示真实数据分布的概率密度，$p_{z}(z)$表示噪声数据分布的概率密度，$G(z_{i})$表示噪声通过生成器生成的数据，它和真实数据$x$形成了一种所谓的对抗关系。$$x_{i}\rightleftharpoons G(z_{i})$$

$D(x)$表示判别器认为$x$为真假的概率，当判别器认为样本越真时，它的值就越接近于1，具体来说：
- $D(x)=1$:当判别器认为输入的样本是来自真实的数据分布
- $D(x)=0$：当判别器认为输入的样本是通过生成器产生的

对于生成器来说，它希望生成的样本让判别器认为是真实的，公式化就是最大化$D(z_{i})$，即$\max \limits_{D}  (D(G(z_{i})))$；而对于判别器来说，它希望的是尽可能分辨出由生成器产生的样本，同时不分错真实的样本，即$\max \limits_{D} (D(x_{i}))$和$\min \limits_{D} (D(G(z_{i})))$。

其中$\max \limits_{D}  (D(G(z_{i})))$和$\min \limits_{D} (D(G(z_{i})))$表示的就是生成器和判别器的对抗过程。为了后面公式的整合和推导，我们将$\max \limits_{D}  (D(G(z_{i})))$和$\min \limits_{D} (D(G(z_{i})))$做一下等价的变换，将其转换成如下的形式：
- $G:$$\min \limits_{D}log(1-D(G(z_{i})))$
- $D:$$\max \limits_{D} logD(x_{i})$、$\max \limits_{D} log(1-D(G(z_{i})))$

其中使用$log(1-D(G(z_{i})))$代替$D(G(z_{i}))$是为了保证$G$在学习的初期就有很大的梯度;而添加$log$是为了方便后面的求导过程。

接着我们将关于$D$、$G$的关系是合并得到：$$\min \limits_{G} \max \limits _{D} logD(x_{i})+log(1-D(G(z_{i})))$$

在学习的过程中作者使用了小批次的随机梯度下降法，故定义$m$为$mini-batch$的大小，下面我们将$mini-batch$中的所有数据相加再求平均，得到$$\min \limits_{G} \max \limits _{D}  \frac {1}{m} \sum_{i=1}^m logD(x_{i})+log(1-D(G(z_{i})))$$

这样就求得了一个$mini-batch$的期望，也就得到了原文中的$V(G,D)$的式子$$\min \limits_{G} \max \limits _{D} V(G,D)=E_{x \sim p_{data}(x) } logD(x_{i})+E_{x \sim p_{z}(z)}log(1-D(G(z_{i})))$$

___
## 算法描述
___
下面是有关算法描述的伪代码<center> 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190329225617520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

基本过程和上面分析公式推导的过程是一致的，其中需要注意几点：
- 这里作者使用的是**小批次随机梯度下降法**（Minibatch stochastic gradient descent）来训练
- 使用**迭代**的数值方法计算
- 作者强调了在学习的过程中，**优化D的K个步骤和优化G的1个步骤交替进行**，这样是保证G变化的足够慢，使得D保持在最佳值附近
- 使用**梯度上升**更新**判别器**，使用**梯度下降**更新**产生器**
- 在基于梯度的方法中，使用了**动量法**这个改进的方法

___
## 全局最优解的证明
___
对抗网络训练到收敛后，最优的判别器满足$$D^{*}_{G}(x) = \frac{p_{data}(x)}{p_{data} (x)+p_{g}(x)}$$
即使得$p_{g}=p_{data}$

证明：对于任给的生成器G，判别器D训练的标准为最大化$V(G,D)$，将$V(G,D)$转换为如下的形式：$$V(G,D)=\int _{x} p_{data}(x)log(D(x))dx+\int_{z}log(1-D(G(z)))dz=\int _{x} p_{data}(x)log(D(x))dx+ p_{g}(x)log(1-D(x))$$
因为任意的$(a,b)\in R^2$且不包含$\{0,0\}$，$(a,b)\in \{0,1\}$，对于函数$y->alog(y)+blog(1-y)$在$\frac {a}{a+b}$处取得最大值，所以当$V(G,D)$取得最大值时，$p_{data}=p_{g}$。

如果将D的训练目标看做是条件概率$p(Y=y|x)$的最大似然估计，当$y=1$时x来自$p_{data}$，当$y=0$时x来自$p_{g}$，则有如下的转换，更容易理解这个结果。<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190330090059871.png)

同时当$p_{data}=p_{g}$时，C(G)取得最小值$-log4$。
___
## 算法收敛性的证明
___

当生成器G给定时，如果判别器D有足够的能力，则它能达到最优的效果，并且通过更新$p_{g}$来提高以下这个判别标准$$E_{x \sim p_{data}} [logD^{*}_{G}(x)]+E_{x \sim p_{g}}[log(1-D^{*}_{G}(x))]$$使得$p_{g}$收敛于$p_{data}$。

证明：将$V(G,D)$记为$V(G,D)=U(p_{g},D)$，把它看做是关于$p_{g}$的函数，也是关于$p_{g}$的凸函数，则该凸函数上确界的一次单数包括到达最大值处的该函数的导数。换言之，如果$f(x)=sup_{\alpha \in A}f_{\alpha}(x)$且对每一个$\alpha$，$f_{\alpha}(x)$是关于x的凸函数，那么如果$\beta=argsup_{\alpha \in A}f_{\alpha}(x)$，则$\partial{f_{\beta}(x)}=\partial{f}$。这等价于给定对应的G和最优的D，计算$p_{g}$的梯度更新，如前面所证，$sup_{D}U(p_{g},D)$是关于$p_{g}$的凸函数且有唯一的全局最优解，所以当$p_{g}$的更新足够小时，$p_{g}$收敛于$p_{data}$。


___
## 实验
___
作者在包括MNIST、TFD和CIFAR-10的数据集上来训练对抗网络。生成器网络使用的激活函数包括修正线性激活（ReLU）和sigmoid 激活，而判别器网络使用maxout激活。Dropout被用于训练判别器网络。

通过对G生成的样本应用高斯Parzen窗口并计算此分布下的对数似然，来估计测试集数据的概率。高斯的$\sigma$参数通过对验证集的交叉验证获得，结果如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190329231127891.png)

该方法估计似然的方差较大且在高维空间中表现不好，但据我们所知却是目前最有效的办法。<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/201903292312486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

用在图像问题中，我们也可以看出生成的图像很接近于真实图像。


___
## 算法的优缺点
___

缺点：
- $p_{data}(x)$只能隐式的表示
- 训练过程中D和G必须很好的同步

优点：
- 不需要马尔科夫链
- 仅使用后向传播来获得梯度
- 学习过程无需推理
- 模型可以融入多种函数
- 生成器只使用从判别器传来的梯度更新G
- 对于分布的表示很尖锐，即使是某些退化的分布

具体的和深度定向图模型、深度无定向图模型、生成自动编码器、对抗模型四种模型的对比如下<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190329231917190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

___
## 总结
___

最后作者给出了未来可以继续进行的工作的某些方向：
- 条件生成模型p(x∣c)可以通过将c作为G和D的输入来获得
- 给定x，可以通过训练一个辅助网络来学习近似推理以预测z。这和wake-sleep算法训练出的推理网络类似，但是它具有一个优势，就是在生成器网络训练完成后，这个推理网络可以针对固定的生成器网络进行训练
- 能够用来近似模拟所有的条件概率p(xS∣xS̸)，其中S是通过训练共享参数的条件模型簇的关于x索引的一个子集。本质上，可以使用生成对抗网络来随机拓展确定的MP-DBM
- .半监督学习：当标签数据有限时，判别网络或推理网络的特征会提高分类器效果
- 效率改善：为协调G和D设计更好的方法或训练期间确定更好的分布来采样z，能够极大的加速训练。

通过阅读后面相关GAN改进的论文我们可以发现，它们很多的工作正是以上面提到的相关思路在做改进。

___
## 参考
___
论文地址： https://arxiv.org/abs/1406.2661

论文源代码： http://www.github.com/goodfeli/adversarial

[个人总结：Generative Adversarial Nets GAN原始公式的得来与推导](https://blog.csdn.net/yyhhlancelot/article/details/83058715)

[Generative Adversarial Nets（译）](https://www.jianshu.com/p/8e1a95c81dfd)

[Generative Adversarial Nets[Vanilla]](https://www.cnblogs.com/shouhuxianjian/p/8182742.html)

[GAN（Generative Adversarial Nets）的发展](https://www.cnblogs.com/huangshiyu13/p/5984911.html)

[Generative Adversarial Nets论文笔记+代码解析](https://blog.csdn.net/wspba/article/details/54582391)

[一周论文 | GAN（Generative Adversarial Nets）研究进展](http://www.sohu.com/a/123700295_465975)

[一文读懂对抗生成学习(Generative Adversarial Nets)[GAN]](https://www.jianshu.com/p/279c959ae8d6)


