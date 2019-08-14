## $ \color{lime}{A\ paper\ a\ day\ keeps\ trouble\ away\!}$

论文地址：https://arxiv.org/abs/1704.02510

论文GitHub：https://github.com/duxingren14/DualGAN

收录： ICCV 2017

![1554540104172](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554540104172.png)

这篇论文和前面看过的《Learning to Discover Cross-Domain Relations with Generative Adversarial Networks》（参见：https://blog.csdn.net/Forlogen/article/details/89003879）和《Image-to-Image Translation with Conditional Adversarial Networks》（参见：https://blog.csdn.net/Forlogen/article/details/89045651）中的内容基本上是相近的。作者也是提出了一种DualGAN的模型，在没有标签数据的前提下，实现在两个不同的域之间的图像转换。整体的算法思想和DiscoGAN、CycleGAN是一致的，并没有什么不同之处，只是名字不一样~~

___

### 算法

所以下面主要介绍一下这篇论文的某些好的地方，其余和上面提到过的两篇论文相同的地方就不赘述了。

先看下它的模型架构：

![1554540557241](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554540557241.png)

这里也是两个生成器和两个判别器，计算判别损失和重构损失。不同之处在于他这里使用的是WGAN中的损失函数Wasseratein loss，而不是标准GAN中使用的交叉熵，它的优点如下：

- 生成模型的收敛性好
- 生成的样本质量高
- 优化过程稳定性好
- 任何地方都是可微的，方便求梯度

因此$D_{A}$ 和 $D_{B}$ 的损失函数定义如下：
$$
l_{A}^d(u,v) = D_{A}(G_{A}(u,z))-D_{A}(v)
\\ l_{B}^d(u,v) = D_{B}(G_{B}(v,z'))-D_{B}(u)
$$
整体损失为：
$$
l^g(u,v)=\lambda_{U}||u-G_{B}(G_{A}(u,z),z')||+\lambda_{V}||u-G_{A}(G_{B}(v,z'),z)||-D_{B}(G_{B}(v,z'))-D_{A}(G_{A}(u,z))
$$
其中$\lambda_{U}$ 和 $\lambda_{V}$ 是两个常参数，取值范围为$[100.0,1,000.0]$ ，同时作者提出，如果U中包含自然的图像，而V中没有时，要使用的$\lambda_{U}$ 小于 $\lambda_{V}$ 。



网络架构和《Image-to-Image Translation with Conditional Adversarial Networks》中的一样，这样即可以抓住图像局部高频的信息，也可以通过重构损失抓住全局的、低频的信息。

___

算法伪代码如下：

![1554541417663](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554541417663.png)

训练的过程和WGAN一样，使用小批次随机梯度下降，并使用RMSProp优化器（有关梯度下降的相关优化方法可参见：https://blog.csdn.net/Forlogen/article/details/88778770）；D的训练轮次是2-4；批大小为1-4；剪裁系数$c$ 取自$[0.01,0.1]$ 。其中$clip(\omega_{A},-c,c),clip(\omega_{B},-c,c)$  这一步的含义待下一篇WGAN在了解。

___

### 实验部分

通过在不同的多个数据及上进行试验，比较DualGAN、GAN 和CGAN的效果差异，评估手段和

《Image-to-Image Translation with Conditional Adversarial Networks》同样相同。通过多次试验证明，DualGAN在大多数的场景下，效果都优于GAN和CGAN，下面给出几张结果图：



白天和黑夜的转换

![1554542062071](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554542062071.png)

照片和素描图的转换

![1554542104994](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554542104994.png)

绘画风格的转变

![1554542133884](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554542133884.png)

但是在图像分割上，DualGAN的效果要差于CGAN，在FACDES->LABEL和AERIAL->MAP两个数据集上都是如此。结果如下：

![1554542282714](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554542282714.png)

![1554542288648](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554542288648.png)

作者认为，可能是因为训练集中没有图像对应信息，很难从中推断出合适的标记。所以最后作者提出能否在训练集中加一小部分带标签的数据来缓解这个问题，具体的工作并没有阐述。

此外在定性评估部分作者提出，DualGAN可能会将像素映射到错误的标签，或将标签映射到错误的纹理/颜色上，这也是一个急需解决的问题。