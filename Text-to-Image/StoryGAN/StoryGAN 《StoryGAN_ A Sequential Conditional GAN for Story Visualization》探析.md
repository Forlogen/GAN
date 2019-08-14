论文地址：https://arxiv.org/abs/1812.02784
GitHub：https://github.com/yitong91/StoryGAN
收录于：CVPR 2019<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623100549171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

之前已经介绍过的很多关于Text-to-Image的相关文章，例如[MirrorGAN](https://blog.csdn.net/Forlogen/article/details/91473574)、[StackGAN](https://blog.csdn.net/Forlogen/article/details/89094148)、[SDGAN](https://blog.csdn.net/Forlogen/article/details/90727750)等，其他相关的paper会在后续时间依次给出总结。而这些Text-to-Image的模型都是基于输入的一句话生成对应的图像，希望生成图像在足够逼真的前提下，同时尽可能的符合输入语句所表达的语义信息。而本文提出的StoryGAN可以实现根据一段连续的描述文本生成序列的对应图像，对于每一张图像来说，它的要求和之前的Text-to-Image的模型一样，但是所有生成图像的集合也要满足整个描述文本的语义一致性，这是之前模型所无法做到的。与之相关的另一个领域的问题是视频生成，但是仍与本文所解决的问题有所差别，在视频生成中需要考虑每一帧的真实性和帧之间过渡的平滑性，而StoryGAN并不需要考虑后一个问题。

整个StoryGAN可以看做是一个**序列化的条件生成对抗网络框架**（sequential conditional GAN framework），模型主要包含一个处理故事流的**深度上下文编码器**（Deep Context Encoder）和**两个Discriminator**（image level和story level）。同时本文的另一个贡献是提供了两个新的数据集**CLEVR-SV**和**Pororo-SV**，实验证明了StoryGAN取得了state-of-the-art的效果。

对于这种全局性的Text-to-Image任务来说，主要有如下的两个困难：
- **如何保持生成的图像序列在表达上保持一致性**，就单句而言，完成的就是一般意义上的text-to-image的工作
- **如何表示故事的逻辑主线**，即随着故事的发展，角色的外观和背景的布局必须以一种连贯的方式呈现

下面通过一个简单的例子理解一下上面的两个问题。如下所示，当输入为“Pororo and Crong are fishing together.Crong is looking at the bucket. Pororo has a fish on hisfishing rod.”时，它包含三个句子，那就需要生成三张图像。我们希望将整个故事的描述和当前所关注的句子一个喂给模型的图像生成器时，它可以生成逼真的图像，最后整体来看又可以保持一致性，即人从直观上认为生成的图像序列和故事描述是一致相符的。<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623102718998.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)
___
### StoryGAN
模型的整体架构如下所示:<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623103554742.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

整体上来看StoryGAN就是一个序列化的生成模型，上图可看做是绿框部分按时间步展开的形式，主要部分为：**Story Encoder、Context Enocder、Image Generator、Image Discriminator和Story Discriminator**，其中Context Encoder中又包含**GRU单元**和**Text2Gist单元**两部分。模型的架构并不难理解，下面就逐个的看一下它们是如何配合完成Story-to-Sequential Images的工作。

在此之前先做一些符号上的规定：
- Story：$S=\{s_{1},s_{2},...,s_{T}\}$，其中$T$表示故事描述的长度，$T$的值是可变的
- 生成图像序列：$\hat{X}=\{\hat{x_{1}},\hat{x_{2}},...,\hat{x_{T}}\}$，其中$\hat{x_{i}}$和$s_{i}$是一一对应的，对应关系可理解为<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623104833356.png)
- 真实图像序列：$X=\{x_{1},x_{2},...,x_{T}\}$
- $h_{o}$：故事描述$S$经过Story Encoder得到的低维向量
- $o_{t}$：每一个单独的句子和前一时刻的$h_{t-1}$经过Context Encoder得到的向量<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623105831697.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

因此如果每个生成图像都符合对应输入句子的语义，那么它们在局部是连贯的，如果生成图像合起来可以描述整个故事，那么它们全局上也是连贯的。
___

### Story Encoder
Story Encoder $E(.)$所做的就是将故事描述$S$随机映射到一个低维的向量空间，得到的向量$h_{0}$不仅包含了$S$全部的信息，同时还作为Context Encoder隐状态的初始值。<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623112201547.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

$h_{0} \sim E(S)=\mathcal{N}(\mu(S), \Sigma(S))$，其中$\mu(S)$为$h_{0}$为所满足分布的均值，$\Sigma(S)$为分布的协方差矩阵，这里作者为了后续计算的效率，将其限制为一个对角阵，即$\boldsymbol{\Sigma}(\boldsymbol{S})=\operatorname{diag}\left(\boldsymbol{\sigma}^{2}(\boldsymbol{S})\right)$，最终$h_{0}$的表达式可写作$h_{0}=\mu(S)+\sigma^{2}(S)^{\frac{1}{2}} \odot \epsilon_{S}, \text { where } \epsilon_{S} \sim \mathcal{N}(0, I)$，其中$\epsilon_{S}$为满足标准正态分布的噪声向量，即StackGAN中**Conditioning Augmentation**的应用。<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019062311294683.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

另外作者为了解决原先故事空间不连续的问题使用了随机采样，不仅在故事的可视化工作中使用了一种更加紧凑的语义表示，而且还为图像的生成过程增加了随机性。同时为了增强条件流行在潜在语义空间上的平滑性，避免生成模式坍缩到单个的生成点上，引入了一个正则化项，即计算生成分布和标准正态分布之间的KL散度<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/201906231400368.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

总体来看，Story Encoder就是想习得一个关于$S$的低维向量。
___
### Context Encoder
在故事的可视化任务重，角色、动作、背景是经常发生变化的
- 如何从上下文信息中捕获到背景的变化？
- 如何组合新的输入语句和随机噪声来生成图像表示角色间的变化，有时这种变化还很巨大？

本文提出了一种基于深度RNN的上下文编码器，它由两个隐藏层组成，低一层是标准的GRU单元，另一层是GRU的变体，作者称之为Text2Gist，如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623135941864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

其中每一层的公式表达如下所示<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623140113500.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

其中$g_{t}$的初始状态$g_{0}$是采样自等距高斯分布（isometric Gaussian distribution）。Text2Gist详细的更新公式如下所示：<cente>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623140439147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

通过Context Encoder在多个时间步的更新，每一时刻都会得到$o_{t}$，将其输入到后续的Image Generator中便可生成对应的图像$\hat{x_{t}}$。

___
### Discriminators
判别器部分存在两个判别器：
- 基于深度神经网络的Image D，保持生成图像的局部一致性，即比较两个三元组$\{s_{t},h_{0},x_{t}\}$和$\{s_{t},h_{0},\hat{x_{t}}\}$
- 基于多层感知机的Story D，保持生成图像的全局一致性

Image D只是用来判别生成图像是否足够真实，因此和其他模型中的并无区别。Story D的原理如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623141430342.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

将输入的语句和相应的生成图像分别降维得到它们的特征向量，然后各自拼接起来送到一个含Sigmoid的全连接网络中，判别生成的图像是否在语义上具有连贯性。如果将图像的特征向量记为$E_{img}(X)=\left[E_{i m g}\left(\boldsymbol{x}_{1}\right), \cdots, E_{i m g}\left(\boldsymbol{x}_{T}\right)\right]$，对应的语句的特征向量记为$E_{t x t}(\boldsymbol{S})=\left[E_{t x t}\left(s_{1}\right), \cdots, E_{t x t}\left(s_{T}\right)\right]$，全局一致性分数的计算公式如下所示：$$D_{S}=\sigma\left(\boldsymbol{w}^{\top}\left(E_{i m g}(\boldsymbol{X}) \odot E_{t x t}(\boldsymbol{S})\right)+b\right)$$其中$w$和$b$是需要学习的参数。通过语句和图像的配对数据，Story D可以同时考虑局部信息和全局信息。
____
综上，StoryGAN的目标函数为$$\min _{\boldsymbol{\theta}} \max _{\boldsymbol{\psi}_{I}, \boldsymbol{\psi}_{S}} \alpha \mathcal{L}_{I m a g e}+\beta \mathcal{L}_{\text {Story}}+\mathcal{L}_{K L}$$
其中$$\begin{aligned} \mathcal{L}_{\text {Image}} &=\sum_{t=1}^{T}\left(\mathbb{E}_{\left(\boldsymbol{x}_{t}, \boldsymbol{s}_{t}\right)}\left[\log D_{I}\left(\boldsymbol{x}_{t}, \boldsymbol{s}_{t}, \boldsymbol{h}_{0} ; \boldsymbol{\psi}_{I}\right)\right]\right.\\ &+\mathbb{E}_{\left(\boldsymbol{\epsilon}_{t}, \boldsymbol{s}_{t}\right)}\left[\log \left(1-D_{I}\left(G\left(\boldsymbol{\epsilon}_{t}, \boldsymbol{s}_{t} ; \boldsymbol{\theta}\right), \boldsymbol{s}_{t}, \boldsymbol{h}_{0} ; \boldsymbol{\psi}_{I}\right)\right)\right] ) \\ \mathcal{L}_{\text {Story}} &=\mathbb{E}_{(\boldsymbol{X}, \boldsymbol{S})}\left[\log D_{S}\left(\boldsymbol{X}, \boldsymbol{S} ; \boldsymbol{\psi}_{S}\right)\right] \\ &+\mathbb{E}_{(\boldsymbol{\epsilon}, \boldsymbol{S})}\left[\log \left(1-D_{S}\left(\left[G\left(\boldsymbol{\epsilon}_{t}, \boldsymbol{s}_{t} ; \boldsymbol{\theta}\right)\right]_{t=1}^{T}\right), \boldsymbol{S} ; \boldsymbol{\psi}_{S}\right)\right) ] \end{aligned}$$

算法描述如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623142400430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

___
## Experiment
实验部分的基准模型为ImageGAN、SVC和SVFN，后两个是StoryGAN的某种简化形式，数据集为CLVER-SV Dataser和Pororo dataset。其中SVC的架构如下所示：<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623142659539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

简单的看下实验结果，对于SSIM值的评估，可以看到StoryGAN的效果更好<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623142807258.png)

在CLVER-SV 数据集上的生成效果，可以看到StoryGAN的结果更接近真实情况<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623142900981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

在Pororo-SV数据集上的生成效果，可以看到StoryGANg同样是取得了最好的效果<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623143008274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

而且当输入语句中的角色姓名改变时，StoryGAN可以很好的把握到这种变化，生成对应的图像<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623143137822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZvcmxvZ2Vu,size_16,color_FFFFFF,t_70)

另外在人类评测中，StoryGAN的结果在多个方面都要更好<center>
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190623143252508.png)
