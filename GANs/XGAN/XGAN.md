论文地址：https://arxiv.org/pdf/1711.05139v6.pdf

![1555944665782](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555944665782.png)

从题目中我们可以看出这是一篇属于Image-to-Image方面的文章，作者提出了一种新的GAN的变体，称为XGAN（GAN架构的形状像一个X），实现了在无监督学习的方式下，完成不同域（domain）的图像之间的转换。在GAN的诸多变体中，类似的有CycleGAN、DualGAN、DiscoGAN以及使用CGAN做Image-to-Image等等很多的方法，作者在实验部分也和CycleGAN进行了对比，显示了XGAN的改进效果。

XGAN中和CycleGAN相比而言，最大的改变就是它使用了语义一致性损失（semantic-consistency loss）从而保留了图像在语义级别的特征信息，而在CycleGAN中关注的是pixel层级的一致性损失，所以XGAN可以保留更高层次的特征信息，生成的图像的结果更加的好。

___

下面我们主要看一下XGAN的模型架构和损失函数，其他导引的部分，具体可见原论文。

首先看一下它的架构，如下所示

![1555945419812](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555945419812.png)

从上图中我们可以大致的看出，XGAN其实就是两个Auto-encoder的组合，通过设置不同的损失项添加约束来达到Image-to-Image的效果。

其中，$D_{1}、D_{2}$表示两个域，$e_{1}、e_{2}$表示两个encoder（可以是任何的CNN压缩编码结构，例如用ResNet或者VGG或者其它类型），$d_{1}、d_{2}$表示两个decoder（类似GAN中Generator的CNN结构），中间的$shared\ Embedding$ 表示共享的特征表达，此外还有两个判别器，一个分类器。而且还可以看出，在encoder的的最后几层和decoder的前几层都使用了权值共享的策略，从而使得在转换的过程中，尽可能地保留图像高级的特征的信息，转换后的结果最大程度的保留输入图像的基本特征，比如在这篇文章所做的真实人脸和卡通人脸的转换中，转换后的人脸的头发、鼻子、眼睛等特征基本上不会变化太大。

模型的架构看起来并不复杂，那么它是如何优于CycleGAN及其他的类似的模型的呢？XGAN优于其他模型的关键在于损失函数的设计，它包含有五个主要的损失项：

- 重建损失项$\mathcal{L}_{rec}$：它和Auto-encoder中的重建损失含义是一样的，即希望encoder编码得到的关于某个域的特征信息输入到decoder中后，得到的重建图像尽量接近于输入图像。它的表达形式如下所示：
  $$
  \mathcal{L}_{r e c, 1}=\mathbb{E}_{\mathbf{x} \sim p_{\mathcal{D}_{1}}}\left(\left\|\mathbf{x}-d_{1}\left(e_{1}(\mathbf{x})\right)\right\|_{2}\right)
  \\\mathcal{L}_{r e c, 2}=\mathbb{E}_{\mathbf{x} \sim p_{\mathcal{D}_{2}}}\left(\left\|\mathbf{x}-d_{2}\left(e_{2}(\mathbf{x})\right)\right\|_{2}\right)
  \\\mathcal{L}_{r e c}=\mathcal{L}_{r e c, 1}+\mathcal{L}_{r e c, 2}
  $$
  其中$x$表述输入样本，$||.||$表示度量重建后的样本和输入样本的差异。

- 域对抗损失$\mathcal{L}_{dann}$ ：它使得$e_{1}$和$e_{2}$学到的嵌入特征信息分布在相同的子空间中，即每个auto-encoder处理后后的图片如果仍可被分类器分辨，则表示编码中包含域信息，而不仅仅是特征信息；反之若不能被分辨，则表示编码内皆为两域共通的特征信息。因此，分类器$c_{dann}$要最大化分类的精度，$e_{1}$和$e_{2}$要同时最小化$\mathcal{L}_{dann}$。它的表达形式如下所示：
  $$
  \begin{array}{l}{\min _{\theta_{e_{1}}, \theta_{e_{2}}} \max _{\theta_{d a n n}} \mathcal{L}_{d a n n}, \text { where }} \\ {\mathcal{L}_{d a n n}=\mathbb{E}_{p_{\mathcal{D}_{1}}} \ell\left(1, c_{d a n n}\left(e_{1}(\mathrm{x})\right)\right)+\mathbb{E}_{p_{\mathcal{D}_{2}}} \ell\left(2, c_{d a n n}\left(e_{2}(\mathrm{x})\right)\right)}\end{array}
  $$
  

- 语义一致性损失$\mathcal{L}_{sem}$：它表示decoder在对两个域进行编码后，在语义特征这个层次上的信息应该一致，从而使得得到的编码具有固定的信息，而且转换后的图像基本特征上和输入是相似的。它的表达形式定义如下：
  $$
  \begin{array}{l}{\mathcal{L}_{s e m, 1 \rightarrow 2}=\mathbb{E}_{\mathbf{x} \sim p_{\mathcal{D}_{1}}}\left\|e_{1}(\mathbf{x})-e_{2}\left(g_{1 \rightarrow 2}(\mathbf{x})\right)\right\|, \text { likewise for } \mathcal{L}_{s e m, 2 \rightarrow 1}} \\ {\|\cdot\| \text { denotes a distance between vectors. }}\end{array}
  $$

- 生成对抗损失$\mathcal{L}_{gan}$ :它和传统的GAN的损失项的含义是一致的，即希望生成更逼真的图像，它的表达形式如下所示：
  $$
  \begin{array}{l}{\min _{\theta_{g_{1 \rightarrow 2}}} \max _{\theta_{1 \rightarrow 2}} \mathcal{L}_{g a n, 1 \rightarrow 2}} \\ {\mathcal{L}_{g a n, 1 \rightarrow 2}=\mathbb{E}_{\mathrm{x} \sim p_{\mathcal{D}}}\left(\log \left(D_{1 \rightarrow 2}(\mathrm{x})\right)\right)+\mathbb{E}_{\mathrm{x} \sim p_{\mathcal{D}_{1}}}\left(\log \left(1-D_{1 \rightarrow 2}\left(g_{1 \rightarrow 2}(\mathrm{x})\right)\right)\right) )}\end{array} \\ \text { Likewise}\ \mathcal{L}_{g a n，2\rightarrow 1}\ is defined\ for\ the\ transformation\ from\ D_{2}\ to\ D_{1}.
  $$

- 指导损失$\mathcal{L}_{teach}$ ：这是一个可选的项，允许使用先验知识来加速模型的训练，可以看作是对习得的Shared Embedding的一种正则化方式，它的表达形式如下所示：
  $$
  \begin{aligned} \mathcal{L}_{\text {teach}}=& \mathbb{E}_{\mathrm{x} \sim p_{\mathcal{D}_{1}}}\left\|T(\mathrm{x})-e_{1}(\mathrm{x})\right\| \\ & \text { where }\|\cdot\| \text { is a distance between vectors. } \end{aligned}
  $$

综合五个主要的损失项得到整体的损失函数为：$\mathcal{L}_{\mathrm{XGAN}}=\mathcal{L}_{r e c}+\omega_{d} \mathcal{L}_{d a n n}+\omega_{s} \mathcal{L}_{s e m}+\omega_{g} \mathcal{L}_{g a n}+\omega_{t} \mathcal{L}_{t e a c h}$ ，其中$\omega$ 为超参数，需要我们设置，因此XGAN也需要不断地调参试优。

___

在实验部分，作者主要完成的是真实人脸和卡通人脸的转换实验，使用的两个数据集是CartoonSet（https://github.com/google/cartoonset）和VGG-Face

![1555948148072](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555948148072.png)

![1555948156091](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555948156091.png)

转换的实验结果如下所示，可以看出效果还是不错的，脸部的基本特征信息都很好的保留了下来

![1555948235700](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555948235700.png)

然后和DTN相比而言，效果要好很多。XGAN 可以更好地捕捉到图像的语义特征信息，生成的图像质量更高，虽然也有一些小问题，但总体效果要优于DTN

![1555948366417](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555948366417.png)

和CycleGAN的实验对比，可以看出CycleGAN生成的图像的质量要差很多

![1555948535642](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1555948535642.png)

此外还通过实验证明了设置的损失项的有效性。

___

XGAN虽然模型的结构很简单，但是通过设置新的损失项，从一个更高的层次把握图像的特征，从而得到了更好的结果。这对于我们的研究工作具有一定的指导意义，当我们思考一个问题时，从不同的高度思考，也许就会有更好的收获。