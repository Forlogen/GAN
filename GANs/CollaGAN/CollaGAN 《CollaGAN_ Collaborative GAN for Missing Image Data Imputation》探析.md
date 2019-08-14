 arxiv：https://arxiv.org/abs/1901.09764

收录：CVPR 2019<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213703715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)


昨天在关于[StarGAN](https://blog.csdn.net/Forlogen/article/details/90116816)的部分提到，它和本文的CollaGAN的模型架构十分的相似，那么它们之间到底有什么不同呢？

首先，我们从题目中可以看出，CollaGAN最直接的目的是用于缺失图像的估计，但是作者指出，对于同一个域（Domain）的图像来说，它们都是满足同样的低维流形，所以关于缺失图像的估计过程就可以转换为图像的处理过程，即相当于实现Image-to-Image的功能。

> the image translation can be considered as a process of estimating the missing image database by modeling the image manifold structure. However, there are fundamental differences between image imputation and image translation.

但是不同于之前的StarGAN的地方在于，StarGAN尽管可以完成多个域之间的图像转换，但是它在实现的过程中每一次只使用一个图像样本，而缺失图像的工作需要使用到干净数据集中其他的数据来具体实现，因此StarGAN就不太适合了。而CollaGAN可以同时接收多个输入，故可以同时利用数据集中的其余数据，通过建模统一的流形结构来实现对于缺失图像数据的估计。而且，CollaGAN生成的图像的效果更好，同样和StarGAN一样，它也只需要一个生成器（Generator，G）,对于内存资源的需求更少。

下面我们通过图直观的看一下CollaGAN和StarGAN在输入数据个数的差异性：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213119199.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

## CollaGAN

CollaGAN不仅模型结构和StarGAN和相似，它的的原理和StarGAN也有很多的类似之处，下面我们具体来看一下它到底是怎样完成缺失图像估计的工作的。首先，给出它的模型结构图<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213452431.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)


假设我们所拥有的数据集总共包含四个域$a,b,c,d$，生成器输出的目标域的假样本记为$x_{a}$，除去输入数据所在的域，其他的域的数据为$\left\{x_{a}\right\}^{C}=\left\{x_{b}, x_{c}, x_{d}\right\}$，上标$C$表示补集，那么G的过程可以表示为:
$$
\hat{x}_{\kappa}=G\left(\left\{x_{\kappa}\right\}^{C} ; \kappa\right)
$$
其中$\kappa$表示目标域，而且关于多个域输入的组合会有很多种，这里是随机选择一个。

### Multiple cycle consistency loss

类似于CycleGAN中的循环一致性损失，因为这里有多个输入，自然对应的就有多个循环一致性损失，原理同样是希望生成的假样本重建后和原来的输入越接近越好，希望可以更好地保留输入图像中的内容数据，当情况如假设所言只有4个域时，其中一个为目标域，那么就需要重建其余三个域的样本，重建后的样本表示如下所示：
$$
\begin{aligned} \tilde{x}_{b|a} &=G\left(\left\{\hat{x}_{a}, x_{c}, x_{d}\right\} ; b\right) \\ \tilde{x}_{c|a} &=G\left(\left\{\hat{x}_{a}, x_{b}, x_{d}\right\} ; c\right) \\ \tilde{x}_{d | a} &=G\left(\left\{\hat{x}_{a}, x_{b}, x_{c}\right\} ; d\right) \end{aligned}
$$
那么使用$L_{1}$范数的损失度量为：
$$
\mathcal{L}_{m c c, a}=\left\|x_{b}-\tilde{x}_{b|a}\right\|_{1}+\left\|x_{c}-\tilde{x}_{c|a}\right\|_{1}+\left\|x_{d}-\tilde{x}_{d | a}\right\|_{1}
$$
推广到一般情况
$$
\mathcal{L}_{m c c, \kappa}=\sum_{\kappa^{\prime} \neq \kappa}\left\|x_{\kappa^{\prime}}-\tilde{x}_{\kappa^{\prime} | \kappa}\right\|_{1}
$$
其中$\tilde{x}_{\kappa^{\prime} | \kappa}=G\left(\left\{\hat{x}_{\kappa}\right\}^{C} ; \kappa^{\prime}\right)$ 。

### Discriminator Loss

类似于StarGAN中的判别器（Discriminator，D）,它不仅需要判断输入的样本是否真实，还需判断输入的样本是输入哪个域的。因此对应的损失可以分为两个部分：adversarial loss和domain classification loss。对于**adversarial loss**，同样为了训练的稳定性，以及为了提高生成样本的质量，这里选择的是[LSGAN](https://blog.csdn.net/Forlogen/article/details/89424640)中的对抗损失项，如下所示：
$$
\mathcal{L}_{g a n}^{d s c}\left(D_{g a n}\right)=\mathbb{E}_{x_{\kappa}}\left[\left(D_{g a n}\left(x_{\kappa}\right)-1\right)^{2}\right]+\mathbb{E}_{\overline{x}_{\kappa | \kappa}}\left[\left(D_{g a n}\left(\overline{x}_{\kappa | \kappa}\right)\right)^{2}\right]
$$

$$
\mathcal{L}_{g a n}^{g e n}(G)=\mathbb{E}_{\tilde{x}_{\kappa | \kappa}}\left[\left(D_{g a n}\left(\tilde{x}_{\kappa | \kappa}\right)-1\right)^{2}\right]
$$

通过最小化上面的两项来分别优化D和G。

对于**domain classification loss**来说，因为输入到D的样本包含生成的假样本和采样的真是样本，因此它也细化为两项$\mathcal{L}_{c l s f}^{r e a l}$和$\mathcal{L}_{c l s f}^{f a k e}$，这里两项使用的都是交叉熵损失（cross entropy loss），具体表达如下所示：
$$
\mathcal{L}_{c l s f}^{r e a l}\left(D_{c l s f}\right)=\mathbb{E}_{x_{\kappa}}\left[-\log \left(D_{c l s f}\left(\kappa ; x_{\kappa}\right)\right)\right]
\\ \mathcal{L}_{c l s f}^{f a k e}(G)=\mathbb{E}_{\hat{x}_{\kappa | \kappa}}\left[-\log \left(D_{c l s f}\left(\kappa ; \hat{x}_{\kappa | \kappa}\right)\right)\right]
$$
当D和G其中某个固定时，通过最小化对应的损失项来分别优化另外不固定的一项。这里需要指出是，为了更合适的训练G，一开始需要使用真实目标域的数据来训练D。

### Structural Similarity Index Loss

另外，为了生成质量更好的结果，作者这里额外引入了一个SSIM损失项：
$$
\mathcal{L}_{m c c-\operatorname{SSIM}, \kappa}=\sum_{\kappa^{\prime} \neq \kappa} \mathcal{L}_{\mathrm{SSIM}}\left(x_{\kappa^{\prime}}, \tilde{x}_{\kappa^{\prime} | \kappa}\right)
$$
它是一个很好的度量图像真实程度的方式，而且本身又是可微的，所以可以方便的使用反向传播进行训练。关于更多关于SSIM的描述，可见原论文。

___

另外，因为CollaGAN只有一个生成器，所以也需要引入Mask Vector。它是一个二进制矩阵，而且与输入图像具有相同的维度，便于联接。每一维都是一个one-hot向量，用来指定具体的目标域。

### 实验

作者通过在MR contrast synthesis、CMU Multi-PIE和RaFD三个数据集上进行实验，证实了CollaGAN的优越性，实验结果如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213557833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213612703.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190512213622373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
___


总的看来，CollaGAN和StarGAN不仅在模型结构很相似，同样在原理上也是类似的，只不过使用了多个重构损失项和不同的Mask vector来实现多个输入的效果。而且，通过实验我们也可以看出，它能做的StarGAN也可以完成，只是效果稍微差一点。

> 仅是个人理解，如有不对的地方还望大家告诉一下~~
