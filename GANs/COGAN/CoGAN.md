论文地址：https://arxiv.org/abs/1606.07536

论文Github：<https://github.com/andrewliao11/CoGAN-tensorflow> 

收录：NIPS2016

![1555720370709](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555720370709.png)

这篇文章中，作者提出了另一个的GAN的变种CoGAN，它通过学习多个域（domain）上的联合分布来实现在无配对数据的情况下的风格转换。CoGAN顾名思义是使用了两个GAN。通过一种权值共享机制（weight-sharing constrain）来限制网络的容量，并且使模型倾向于学习不同域的联合分布，而不是各个域的边缘分布的乘积。通过多个实验证明，它可以成功的学习在没有任何配对数据下的域之间的联合分布，在域适应（domain adaption）和图像转换（image transformation）方面效果很好。

___

CoGAN的架构如下所示：

![1555720987929](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555720987929.png)

它使用了两个标准的GAN，每个GAN中只有一个生成器和一个判别器，生成器之间前几层实现权值共享，判别器的后几层之间实现权值共享。

___

下面看一下它的objective function是怎么设置的。

#### Generator

假设$x_{1}$和$x_{2}$是采样自两个不同域的边缘分布的样本，分别满足$x_{1} \sim p_{x_{1}}$和$x_{2} \sim p_{x_{2}}$，$g_{1}$和$g_{2}$分别代表上图所示的$GAN_{1}$和$GAN_{2}$，它们接收一个随机的噪声向量$z$，然后将其映射到分别具有和$x_{1}$和$x_{2}$有相同支持的图像。$g_{1}$和$g_{2}$生成的样本记为$g_{1}(z)$和$g_{2}(z)$ ，分别满足$g_{1}(z) \sim P_{G_{1}}$$g_{2}(z) \sim P_{G_{2}}$ ，那么将$g_{1}$和$g_{2}$通过多层感知器实现，生成的样本可以表示为：
$$
g_{1}(\mathrm{z})=g_{1}^{\left(m_{1}\right)}\left(g_{1}^{\left(m_{1}-1\right)}\left(\ldots g_{1}^{(2)}\left(g_{1}^{(1)}(\mathrm{z})\right)\right)\right), \quad g_{2}(\mathrm{z})=g_{2}^{\left(m_{2}\right)}\left(g_{2}^{\left(m_{2}-1\right)}\left(\ldots g_{2}^{(2)}\left(g_{2}^{(1)}(\mathrm{z})\right)\right)\right)
$$
对于生成器来说，它的第一层学习到的是输入图像的高级语义部分，而最后一层学习到的是图像的底层的一些细节部分；对于判别器来说，正好相反。

为了使得两个域中的图共享相同的高层语义特征，在$g_{1}$和$g_{2}$的前$K$层实现权值共享限制，即$\theta_{g_{1}^{(i)}}=\theta_{g_{2}^{(i)}},i=1,2,...,k$，这个约束迫使$g_{1}$和$g_{2}$中以相同的方式解码高级语义，但最后一层没有任何约束，这使得它们又可以学习到各自输入的底层特征，来欺骗各自的判别器。

在实验中作者发现生成器的效果和权值共享的层数多少呈正相关关系。

___

#### Discriminator

假设$f_{1}$和$f_{2}$分别代表两个discriminator，同样使用多层感知器实现，表示为：
$$
f_{1}\left(\mathrm{x}_{1}\right)=f_{1}^{\left(n_{1}\right)}\left(f_{1}^{\left(n_{1}-1\right)}\left(\ldots f_{1}^{(2)}\left(\mathrm{f}_{1}^{(1)}\left(\mathrm{x}_{1}\right)\right)\right)\right), f_{2}\left(\mathrm{x}_{2}\right)=f_{2}^{\left(n_{2}\right)}\left(f_{2}^{\left(n_{2}-1\right)}\left(\ldots f_{2}^{(2)}\left(f_{2}^{(1)}\left(\mathrm{x}_{2}\right)\right)\right)\right)
$$
它将输入到它的样本映射到一个概率分数，从而判别该样本有多大可能性来自于真实的数据分布。在判别器中，第一层提取低级特征，而最后一层提取高级特征。而由于在生成器中的约束，输入的图像样本在两个不同的域中共享了高级语义特征，因此我们强制$f_{1}$和$f_{2}$的最后一层实现权值共享，即${\theta}_{f_{1}^{\left(n_{1}-i\right)}}={\theta}_{f_{2}^{\left(n_{2}-i\right)}} $$，i=1,2,...,l$。

在后面的实验中作者发现，判别器中权值共享的层数多少与效果无关，但是这里仍然使用它是为了减少网络中的参数。

___

从上面的分析中我们可以看出，CoGAN的$G$不同于传统的CNN，它实现的是一种上采样的操作，从而来放大图像。如下所示，$G$的前几层解码的是高层语义信息，最后解码的是底层纹理信息； $D$与传统CNN则类似，低层解码底层纹理信息，高层解码高层语义信息。因此，通过共享$G$的前几层的权值，可以保证生成的两个域的数据在高层语义特征上类似，而后几层不加约束， 保证了两个域间有各自的特征。

传统的CNN的下采样操作：

![](E:\软件\Data\深度学习\GAN\Classic-GANs\COGAN\no_padding_no_strides_transposed.gif)

CoGAN中$G$的上采样操作：

![](E:\软件\Data\深度学习\GAN\Classic-GANs\COGAN\padding_strides_odd_transposed.gif)

___

#### learning

因此CoGAN就表示成一个受限的博弈游戏过程，对应的objective function可以写为：
$$
\begin{aligned} V\left(f_{1}, f_{2}, g_{1}, g_{2}\right) &=E_{\mathbf{x}_{1} \sim p_{\mathbf{X}_{1}}}\left[-\log f_{1}\left(\mathbf{x}_{1}\right)\right]+E_{\mathbf{Z} \sim p_{\mathbf{Z}}}\left[-\log \left(1-f_{1}\left(g_{1}(\mathbf{z})\right)\right)\right] \\ &+E_{\mathbf{x}_{2} \sim p_{\mathbf{X}_{2}}}\left[-\log f_{2}\left(\mathbf{x}_{2}\right)\right]+E_{\mathbf{z} \sim p_{\mathbf{Z}}}\left[-\log \left(1-f_{2}\left(g_{2}(\mathbf{z})\right)\right)\right] \end{aligned}
$$
我们的目标是在给定约束条件下的最大最小优化问题：
$$
\begin{aligned} V\left(f_{1}, f_{2}, g_{1}, g_{2}\right) &=E_{\mathbf{x}_{1} \sim p_{\mathbf{X}_{1}}}\left[-\log f_{1}\left(\mathbf{x}_{1}\right)\right]+E_{\mathbf{Z} \sim p_{\mathbf{Z}}}\left[-\log \left(1-f_{1}\left(g_{1}(\mathbf{z})\right)\right)\right] \\ &+E_{\mathbf{x}_{2} \sim p_{\mathbf{X}_{2}}}\left[-\log f_{2}\left(\mathbf{x}_{2}\right)\right]+E_{\mathbf{z} \sim p_{\mathbf{Z}}}\left[-\log \left(1-f_{2}\left(g_{2}(\mathbf{z})\right)\right)\right] \end{aligned}
$$
这样两个生成器之间通过权值共享学习高级的语义特征，可以实现图像在不同域的转换；同时分别保留了自己的底层的特征细节，使它们可以经过学习生成更接近真实的样本来达到骗过判别器的效果。而对于判别器来说，它们只是在最后几层实现权值共享，也不影响对于输入图像底层细节特征的把握，因此也可以最大程度地判别输入的来源。

___

#### experiments

作者主要是和CGANs进行了效果的对比，显示了CoGAN的效果要优于它。下面给出一些实验结果的图：

使用CoGAN生成具有不同属性的人脸图像

![1555724184445](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555724184445.png)

使用CoGAN生成颜色和深度图像

![1555724207638](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555724207638.png)

更多的实验细节部分和相关的网络架构可见原论文的附录部分。