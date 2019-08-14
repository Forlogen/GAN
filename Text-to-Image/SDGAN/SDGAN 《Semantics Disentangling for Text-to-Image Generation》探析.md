论文地址：<https://arxiv.org/abs/1904.01480v1>

收录于：CVPR 2019<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601093158848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

这篇文章属于Text-to-Image一类，它所解决的主要任务是如何根据文本的描述生成相应的图像，不仅要求生成的图像要清晰真实，而且更要求其符合给定的文本描述。类似的GAN模型有StackGAN（[《StackGAN:Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks》探析](https://blog.csdn.net/Forlogen/article/details/89094148)）、StackGAN++、AttenGAN……，已经同样是收录于CVPR2019上的一篇StoryGAN，当然解决同样任务的模型还有很多，更多可见

> [arbitrary-text-to-image-papers](http://bbs.cvmart.net/topics/356/arbitrary-text-to-image-papers-tu-xiang-wen-ben-sheng-cheng-lun-wen-hui-zong)

首先看一下本文所提出的SDGAN的效果如何<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601093938267.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

如上图所示，SDGAN和之前的StackGAN、AttenGAN进行了对比，从生成的图像可以直观的看出SDGAN的结果更清晰，同时与描述文本更相符，那么它是做了哪些工作才得到这么好的效果呢？首先看下题目：**Semantics Disentangling for Text-to-Image Generation**，将其翻译过来为用于**用于Text-to-Image的语义分解**，从中我们可以知道它是基于文本的语义来改进Text-to-Image的工作，进一步可以猜测它应该是考虑文本的高级语义和低级语义，SDGAN具体是如何做的，下面具体的来看一下。

作者指出由于对于同一张图像来说，不同的人会给出不同的描述，它们自然包含了对于图像主要信息的描述，同样也充满了多样性和个性化，因此如何从中提取出一致性的语义信息，同时保留描述的多样性和其中的细节信息，成为了Text-to-Image任务的一大难题。
___

在正式理解SDGAN模型之前，我们需要补充一些相关的基础知识，主要是Siamese structure network、Batch Normalization。
### Siamese structure network
Siam是古代对于泰国的称呼，可译为暹(xian)罗,Siamese自然可译为暹罗人或泰国人。但Siamese structure network中的Siamese可译为孪生、连体

> 十九世纪泰国出生了一对连体婴儿，当时的医学技术无法使两人分离出来，于是两人顽强地生活了一生，1829年被英国商人发现，进入马戏团，在全世界各地表演，1839年他们访问美国北卡罗莱那州后来成为“玲玲马戏团” 的台柱，最后成为美国公民。从此之后“暹罗双胞胎”（Siamese twins）就成了连体人的代名词，也因为这对双胞胎让全世界都重视到这项特殊疾病。

**Siamese network**最早在2005年Yann Lecun提出，它可以看做是一种相似性度量方法，而它所主要解决的是one-shot（few-shot）的任务。one-shot任务是指在现实的场景中，数据集中的样本可能类别数很多，但是每个类别所包含的样本的数量很少，甚至极端情况下只有一张，在这样的情况下，如果使用传统的分类模型去做，往往得到的效果都不会太好，因为传统的模型依赖于**大量有标注**的样本。而one-shot希望做的就是使用极少的样本也可以得到不错的效果。

Siamese network试图从数据中去学习一个相似性度量，然后用这个习得的度量去比较和匹配新的未知类别的样本。它的模型架构如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601102458745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

如果只是在pixel空间中进行相似性度量显然不合适，因此Siamese network通过某个映射函数将输入的样本映射到一个目标空间中，然后在目标空间中使用一般的距离度量方式进行相似度比较，希望同类的样本的距离应该相近，而不同类别的样本距离应较远。如上图所示，$G_{W}(X)$表示需要学习的映射函数，在Siamese network中只要求其可微，并不附加其他的限制条件，其中的参数$W$的求解就是主要的工作。对于输入$X_{1}$和$X_{2}$来说，当它们是同类别的样本时，相似性度量$E_{W}\left(X_{1},X_{2}\right)=\left\|G_{W}\left(X_{1}\right)-G_{W}\left(X_{2}\right)\right\|$的值较小；当它们是不同类别的样本时，$E_{W}\left(X_{1}, X_{2}\right)=\left\|G_{W}\left(X_{1}\right)-G_{W}\left(X_{2}\right)\right\|$的值较大。因此，在训练集上使用成对的样本进行训练，输入同类别时最小化损失函数$E_{W}\left(X_{1}, X_{2}\right)$，而输入是不同类别时最大化$E_{W}\left(X_{1}, X_{2}\right)$。

在模型中左右两个网络共享参数，因此可以看做两个完全相同的网络，它们的输出为低维空间中的$G_{W}(X_{1})$和$G_{W}(X_{2})$，然后使用能量函数$E_{W}(X_{1},X_{2})$进行比较。如果假设损失函数只与输入$X_{i}$和参数$W$有关，它的形式如下所示：$$\begin{aligned} \mathcal{L}(W) &=\sum_{i=1}^{P} L\left(W,\left(Y, X_{1}, X_{2}\right)^{i}\right) \\ L\left(W,\left(Y, X_{1}, X_{2}\right)^{i}\right) &=(1-Y) L_{G}\left(E_{W}\left(X_{1}, X_{2}\right)^{i}\right) \\ &+Y L_{I}\left(E_{W}\left(X_{1}, X_{2}\right)^{i}\right) \end{aligned}$$其中$(Y,X_{1},X_{2})$表示第$i$个样本，是由一组配对图片和一个标签（$Y=0$表示同类别$Y=1$表示不同类别，）组成的，其中$L_{G}$是只计算相同类别对图片的损失函数，$L_{I}$是只计算不相同类别对图片的损失函数。$P$是训练的样本数。通过这样分开设计，可以达到当我们要最小化损失函数的时候，可以减少相同类别对的能量，增加不相同对的能量。

> [1] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity metric discriminatively, with application to face verification. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pages 539–546. IEEE, 2005. 
[2] Mohammad Norouzi, David J. Fleet, Ruslan Salakhutdinov, Hamming Distance Metric Learning, Neural Information Processing Systems (NIPS), 2012.

### Batch Normalization
BN是优化神经网络训练一个很重要的工具，大家应该都比较熟悉了，这里只给出一个简单的总结，如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601104926404.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

这个过程可以表示为$$\mathrm{BN}(x)=\gamma \cdot \frac{x-\mu(x)}{\sigma(x)}+\beta$$

此外对BN的改进有**Conditional Batch Norm**，可以将其看作是在一般的特征图上的缩放和移位操作的一种特例，它的表示形式如下所示$$\operatorname{BN}(x | c)=\left(\gamma+\gamma_{c}\right) \cdot \frac{x-\mu(x)}{\sigma(x)}+\left(\beta+\beta_{c}\right)$$

___

## SDGAN
模型架构如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601105712988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

整个模型可以看作主要由$Siamese mechanism + SCBN$组成，其中使用Siamese mechanism在判别器（Discriminator）中学习高层次的语义一致性，使用SCBN来发现不同形式的底层语义。这样既可以提取出语义的一致性部分，由可以保留语义的多样性和细节部分。

模型的主要部分为：
- **text encoder**：用于提取描述文本中的特征表示，文中采用的双向的LSTM，其中$w_{t}$表示第$i$个词的特征向量，$\overline{\mathcal{s}}$表示整个句子的特征向量；
- **hierarchical generative adversarial subnets**：用于图像的生成，由多对生成器（Generator，G）和判别器（Discriminator，D）组成，实现从低分辨到高分辨率的逐级过渡，每个阶段都由一个$G$和一个$D$配对组成，$G$生成的图像$D$都会给出判别真假的结果。而且它的输入不只是$\overline{\mathcal{s}}$，还由一个采样自标准正态分布的噪声向量$z$，使得对于很相近的描述，模型也可以生成不同的图像；其中$G$如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601114909263.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
- **adversarial discriminators**：用于每阶段生成图像真假的判别

此外还有一个很重要的部分就是**Contrastive Loss**。对于两个语义上相近的描述性文本生成的图像也应该类似，否则生成的图像应相差较远。因此可采用contrastive loss来提取成对的描述性文本输入中的语义信息，损失函数如下所示：$$L_{c}=\frac{1}{2 N} \sum_{n=1}^{N} y \cdot d^{2}+(1-y) \max (\varepsilon-d, 0)^{2}$$其中$d=||v_{1}-v_{2}||$表示两个特征向量之间的距离，$y$表示输入的描述文本都是是针对于同一图像，$y=1$表示相同，$y=0$表示不同。$N$表示特征向量的维度，这里取256，$\epsilon$取1.0。

通过使用Contrastive Loss来最小化来自同一图像描述的生成图像之间的距离和最大化来自不同图像描述的生成图像之间的距离来优化模型。同时为了避免生成无意义的结果，最终采用的形式如下所示$$L_{c}=\frac{1}{2 N} \sum_{n=1}^{N} y \max (d, \alpha)^{2}+(1-y) \max (\varepsilon-d, 0)^{2}$$其中$\alpha=0.1$同样是用来避免生成太过于相近的假样本。

 Semantic-Conditioned Batch Normalization，SCBN：是利用自然语言描述中的语言线索（linguistic cues）来调节条件批处理归一化，主要目的是增强生成网络特征图的视觉语义嵌入。它使语言嵌入能够通过上下缩放、否定或关闭等方式操纵视觉特征图，如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601115329305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

SCBN可以从输入中获取到语句级和词级两个层次上的语言线索，具体内容可见论文。自我感觉这是一个很好的工具，如果后面作者放出源代码，再对其进行详细的学习。
___
## 实验
通过再CUB和MS-COCO两个数据集上进行实验，证明了SDGAN优于已有的模型<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601115903421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

进一步通过实验证明了Siamese mechanism和SCBN的有效性<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601120034402.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601120111608.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

最后显示了SDGAN可以对描述文本中小的变化做出相应的改变<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190601120208281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

___
## 总结
整体来看，本文所使用的模型架构和新提出的SCBN具有一定的吸引力，为后面Text-to-Image 任务的解决提供了新的思路。

## 参考

> https://www.researchgate.net/publication/4156225_Learning_a_similarity_metric_discriminatively_with_application_to_face_verification?enrichId=rgreq-41bb86e9729b1d866d0194963b0d4747-XXX&enrichSource=Y292ZXJQYWdlOzQxNTYyMjU7QVM6MTAxMzE1NTUyMjE5MTQyQDE0MDExNjY5MTg0MTE%3D&el=1_x_3&_esc=publicationCoverPdf
> 
> https://www.jianshu.com/p/92d7f6eaacf5
> 
> https://www.cnblogs.com/bentuwuying/p/8186364.html
> 

