论文地址：https://arxiv.org/pdf/1809.11096.pdf

论文Github：https://github.com/AaronLeong/BigGAN-pytorch

收录：ICLR 2019

![1556006629608](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1556006629608.png)

BigGAN实现了对之前所有有关生成高分辨率的GAN变体的超越，生成的图像真的是太真实了，太秀了！

![1556006750964](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1556006750964.png)

文章中虽然没有太多的数学公式和方法的改变，但是借助强大的计算力，把生成的结果质量提高了一个很高的层次，512块TPU啊，我连块GPU都么得！！默默的留下了穷人的眼泪，人家做科研靠头脑加设备，我们做科研全靠吹水，😔

文章大部分的内容都是在阐释作者的思路，具体的可以参考：

1. 论文的主要内容

   > https://cloud.tencent.com/developer/article/1375746
   >
   > https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/83026880
   >
   > https://blog.csdn.net/qq_14845119/article/details/85619705

2. 有关BigGAN的报道

   > https://www.jianshu.com/p/b36c20df2586
   >
   > http://www.sohu.com/a/273780404_473283
   >
   > http://www.sohu.com/a/275014717_129720
   >
   > http://www.sohu.com/a/257335499_473283
   >
   > http://www.chainske.com/a/show.php?itemid=5028
   >
   > https://www.sohu.com/a/257110847_610300

___

下面主要记录一下其中不太理解的一些做法，主要包括谱归一化（Spectral Normalization）、截断技巧（Truncation Trick）和正交正则化（orthogonal regularization）

### Spectral Normalization

> https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/87220341
>
> https://www.jiqizhixin.com/articles/2018-10-16-19
>
> https://www.jianshu.com/p/d6224fee3365
>
> https://blog.csdn.net/StreamRock/article/details/83590347
>
> https://blog.csdn.net/songbinxu/article/details/84581248

在《Spectral Normalization for Generative Adversarial Networks》（https://arxiv.org/pdf/1802.05957.pdf） 这篇文章中，作者对于谱归一化做了详细介绍，等读完后再记录。

___

### Truncation Trick

之前GAN的生成的输入噪声采样自某个先验分布z，一般情况下都是选用标准正态分布 N(0,I) 或者均匀分布 U[−1,1]。所谓的“截断技巧”就是通过对从先验分布 z 采样，通过设置阈值的方式来截断 z 的采样，其中超出范围的值被重新采样以落入该范围内。这个阈值可以根据生成质量指标 IS 和 FID 决定。

我们可以根据实验的结果好坏来对阈值进行设定，当阈值的下降时，生成的质量会越来越好，但是由于阈值的下降、采样的范围变窄，就会造成生成上取向单一化，造成生成的多样性不足的问题。往往 IS 可以反应图像的生成质量，FID 则会更假注重生成的多样性。

例如在文中作者也给出了使用截断技巧的实验结果图，其中从左到右，阈值=2，1.5，1，0.5，0.04

![1556070368336](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1556070368336.png)

从结果可以看出，随着截断的阈值下降，生成的质量在提高，但是生成也趋近于单一化。所以根据实验的生成要求，权衡生成质量和生成多样性是一个抉择，往往阈值的下降会带来 IS 的一路上涨，但是 FID 会先变好后一路变差。

___

### orthogonal regularization

同时作者还发现，在一些较大的模型中嵌入截断噪声会产生饱和伪影，如下所示

![1556070512466](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1556070512466.png)

因此，为了解决这个问题，作者通过将$G$调节为平滑来强制执行截断的适应性，以便让$z$ 的整个空间将映射到良好的输入样本。为此，文中引入了正交正则化，直接强制执行正交性条件
$$
R_{\beta}(W)=\beta\left\|W^{\top} W-I\right\|_{\mathrm{F}}^{2}
$$
其中$W$是权重矩阵，$\beta$是超参数。文中为了放松约束同时实现所需的平滑度，从正则化中删除了对角项，旨在最小化滤波器之间的成对余弦相似性，但不限制它们的范数
$$
R_{\beta}(W)=\beta\left\|W^{T} W \odot(1-I)\right\|_{F}^{2}
$$
其中 $1$表示一个矩阵，所有元素都设置为 1

