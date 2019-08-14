论文地址：https://arxiv.org/abs/1703.02291v2

论文Github：https://github.com/zhenxuan00/triple-gan

收录：NIPS 2017

![1555726088213](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555726088213.png)

在《Coupled Generative Adversarial Networks》中，作者使用了两个GANs实现了在无配对数据的的情况下，实现不同的domain之间的高层次的特征学习。除了couple GAN，类似的还有这篇文章所讲的TripleGANs，但它并不是直面理解的使用了三个GAN，而是在GAN中添加了一个分类器（classifier）。

___

#### 背景

在这篇文章发表之前，在半监督学习方面，GANs已经取得一些不错的成果，但是它们存在两个问题：

- 生成器$G$和判别器$D$无法各自都达到最优
- 生成器$G$无法掌握生成样本中的语义信息

对于第一个问题，有人使用了feature matching的技术，但是它在分类方面效果不错，但无法产生逼真的图像；也有人使用了minibatch discrimination的技术，但它可以产生逼真的图像，却无法同时在分类方面做的很好。

对于第二个问题，因为生成样本中的语义特征对于之后的应用很关键，所以从中提取出有意义的语义特征也是一件很重要的事情，为了解决这个问题，在其他的GAN的变种中也有专门的解决方案。

作者认为这两个问题的出现，主要是由于在传统的GAN中，$G$、$D$双方博弈的过程使得$D$无法同时兼容两个角色，既判别输入的样本是来自真实数据分布还是生成器，同时预测数据的类标签。这样的做法只能关注到数据的一部分信息，即数据的来源，而无法考虑到数据的类标签信息。

为了解决这个问题，在标准的GAN的基础上引入了分类器$C$，这样它就有了三个部分$G、D、C$，因为取名TripleGAN。G和C分别对图像和标签之间的条件分布进行特征描述，而D只需要专注于判别输入样本image-label的来源。

通过这样的做法，除了解决上面提出的两个问题，TripleGAN还有如下的优势：

- 可以实现很好的分类效果
- 通过在潜在的条件类空间中插值，实现输入样本在类和风格样式的分解，并在数据空间中平滑传输

___

#### 整体理解

TripleGAN的结构如下所示：

![1555728930996](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555728930996.png)

其中$R$表示拒绝，$A$表示接受,$CE$表示监督学习的交叉熵损失。$p_{g}$、$p_{c}$和$p$分别是由生成器、分类器和真实数据生成过程定义的分布，$CE$是保证它们之间一致性的无偏正则化。

上图中的$G$和$C$是两个条件网络，$C$为真实的图像数据生成伪标签，$G$给真实标签对应的假的图像数据，$D$ 只用于判别输入的data-label是否来自真实的带标签的数据集。通过$C$和$G$之间不断地对抗学习，最终可以同时得到不错的$C$和$G$，而且$D$可以从$C$得到关于无标签数据的标签信息，迫使$G$生成正确的image-label数据。这样就解决了其他GANs无法解决的两个问题，并通过理论分析和实验分析，验证了TripleGAN最后可以学习到一个好的分类器和一个好的条件生成器。

___

#### 理论分析

TripleGAN的目标不仅是可以为无标签的数据正确的预测对应的类标签，而且还可以生成真实的data-label数据。

> $$
> P(x,y)=P(x)P(y|x)\\ P(x,y)=P(y)P(x|y)
> $$

首先我们先对TripleGAN中的各部分做一些假设：

- $C$：描述了条件分布$P_{c}(x,y) \approx P(y|x) $
- $G$：描述了条件分布$P_{G}(x,y) \approx P(x|y)$
- $D$：判别$(x,y)$是否来自真是数据分布$P(x,y)$

理想的博弈的最优均衡点出现在，当$C$和$G$定义的条件分布都收敛于真实数据分布 $p(x,y)$ 的情况下。

- $p(x)、p(y)$：分别表示真实数据集中的边缘分布
- $x$：$x = G(y,z)$，表示生成器生成的假样本
- $p_{c}(x,y)$：$p_{c}(x,y)=p(x)p(y|x)$，表示分类器生成的data-label数据
- $p_{g}(x,y)$：$p_{g}(x,y)=p(y)p(x|y)$，表示生成器生成的data-label数据
- $p(x,y)$：表示从真实数据分布中采样的data-label数据

将$p_{c}(x,y)$、$p_{g}(x,y)$、$p(x,y)$都输入到$D$中，让其进行判别。那么三者之间的对间过程如下所示：
$$
\begin{aligned} \min _{C, G} \max _{D} U(C, G, D)=& E_{(x, y) \sim p(x, y)}[\log D(x, y)]+\alpha E_{(x, y) \sim p_{c}(x, y)}[\log (1-D(x, y))] \\ &+(1-\alpha) E_{(x, y) \sim p_{g}(x, y)}[\log (1-D(G(y, z), y))] \end{aligned}
$$
其中$\alpha$为平衡生成和分类重要性的超参数，本文设置$\alpha=\frac{1}{2}$。当且仅当$p(x,y)=(1-\alpha)p_{g}(x,y)+\alpha p_{c}(x,y)$时，才能到达均衡点。但是很难确保当$p(x,y)=p_{g}(x,y)=p_{c}(x,y)$ 时为唯一的全局最优点，为此在$C$中引入了standard supervised loss，定义如下：
$$
\mathcal{R}_{\mathcal{L}}=E_{(x, y) \sim p(x, y)}\left[-\log p_{c}(y | x)\right]
$$
它相当于衡量$p_{c}(x,y)$和$p(x,y)$ 之间的KL散度，因为只要$C$和$G$中有一个趋近于数据的真实分布，另一个也会同样趋近于真实数据分布，所以这里选择一个方向即可。因此objective function转换成如下的形式：
$$
\begin{aligned} \min _{C, G} \max _{D} \tilde{U}(C, G, D)=& E_{(x, y) \sim p(x, y)}[\log D(x, y)]+\alpha E_{(x, y) \sim p_{c}(x, y)}[\log (1-D(x, y))] \\ &+(1-\alpha) E_{(x, y) \sim p_{g}(x, y)}[\log (1-D(G(y, z), y))]+\mathcal{R}_{\mathcal{L}} \end{aligned}
$$
文中证明了$\tilde{U}$对于$C$和$G$有唯一的全局最优点。

> 在实际操作中，作者使用$C$为一些无标签数据生成的伪标签后的data-label数据做为正样本输入到D中，这使得$C$收敛的更快，且$p_{c}$和$p$更接近。

整体的算法描述如下所示：

![1555732455003](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555732455003.png)

更多有关理论方面的证明，可见原论文的3.2和附录部分，很容易理解。

____

#### 实验

作者在MNIST、SVHN、CIFAR0三个数据集上做了实验，分别和其他的相关算法进行比较，显示了TripleGAN分类的效果明显由于其他的算法，甚至好于ImprovedGAN。

![1555749074580](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555749074580.png)

而且当带标签数据减少时，TripleGAN的分类效果也要优于ImprovedGAN。

![1555749168837](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555749168837.png)

生成实验中，在MNIST上的生成效果优于使用feature matching的ImprovedGAN

![1555749304833](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555749304833.png)

在CIFAR10上生成的图像的效果也很好

![1555749348361](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555749348361.png)

____

#### 总结

TripleGAN相比于其他的GANs，在数据方面而言，它不仅是可以生成效果较好的图像，同时还利用了图像的类标签信息。因此可以使用TripleGAN中的生成器，为指定的类标签生成样本；也可以使用其中的分类器，为图像打标签。综合来看，使用TripleGAN可以丰富训练的数据集，减少人工标注的开销。