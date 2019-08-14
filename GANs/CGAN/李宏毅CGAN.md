关于CGAN的论文解读可见：https://blog.csdn.net/Forlogen/article/details/88980919

李宏毅老师的主页：http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html

B站视频：https://www.bilibili.com/video/av24011528?from=search&seid=4633574616752274163

通过阅读CGAN的原始论文，我们可以看到它对标准GAN的改进，以及它的一个应用场景。今天再看李宏毅老师是如何解读CGAN的，具体的PPT和视频可于主页观看、下载。

在CGAN中我们希望可以实现类似Text-to-Image的功能，给定一段关于图像的描述，通过GAN就可以生成相应的图像。如果将其看做传统的有监督的学习的话，将是将一段描述输入到一个神经网络中，完了输入一张图象，我们希望它与真实的描述越接近越好。比如我们输入"a dog is running"，输入的图像就最好应该是一只奔跑的小狗。

![1554341720061](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554341720061.png)

但是这存在一个问题，实际中对于同一物体的观察，通过不同的角度，我们看到的图像是不同的。比如火车，有侧面看飞速前进的火车，有正面看迎面而来的火车……如果我们只是告诉G，生成一张火车的图像，那么它可能就会生成正面火车的图像和侧面火车图像的混合，得到的结果就是模糊不清的。

![1554341893973](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554341893973.png)

那么如何在GAN中中实现Text-to-Image的功能呢？首先容易想到的就是，在b标准GAN的G的输入中添加一个条件c，它描述了要生成什么图像，然后按照GAN的训练方法进行训练，最后输出一个分数。但是这样会有一个很大的问题，慢慢的D只会关注生成的图像是否真实，完全忽略掉条件c的存在，这和标准的GAN就没有区别了。

![1554342684689](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554342684689.png)

为了解决这个问题，我们需要将条件c做为G和D共同的的输入项，这样D最后判别的结果不仅会考虑图像的真实性，同时也会考虑它是否符合条件的描述。比如我们的条件是train，那么D不仅会给不像火车的生成图像一个很低的分数，也会给即使很真实但是不是不是火车的图象一个很低的分数，这就实现了CGAN的基本思想。

![1554342864456](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554342864456.png)

在CGAN的架构中，通常使用的是如下的这一种，将x和c分别输入到一个Network中，然后再将其输出共同输入到另一个Network中给出一个分数。

![1554343054847](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554343054847.png)



同时也存在另一种架构，它将生成的图像传入到一个Network中，先判断它的真实性，然后将这个Network的输入和条件c一起传入到另一个Network中，判断它是否符合条件。这样的话，我们就可以更明显的理解D判别的结果。

![1554343176144](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554343176144.png)

另外还有一种GAN变体StackGAN，它的基本思想是类似与CGAN，不过它先生成一种小的图像，比如64*64，然后将其和条件继续送到StackGAN中，生成一个更大尺寸的图像。具体的论文可见：https://arxiv.org/abs/1612.03242

![1554343362034](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554343362034.png)

另外的一篇论文中也使用CGAN的基本思想，提出了Image-to-Image的功能实现方法。这也是后面即将看的一篇论文。

论文：https://arxiv.org/abs/1611.07004

代码：https://phillipi.github.io/pix2pix/



![1554343657401](C:\Users\dyliang\AppData\Roaming\Typora\typora-user-images\1554343657401.png)