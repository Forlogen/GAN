
**cource link: https://github.com/lzhbrian/arbitrary-text-to-image-papers.git **

# arbitrary-text-to-image-papers

A collection of arbitrary kinds of text to image papers, organized by [Tzu-Heng Lin](https://lzhbrian.me) and [Haoran Mo](https://github.com/MarkMoHR).

Papers are ordered in arXiv first version submitting time (if applicable).

Feel free to send a PR or an issue.



**TOC**

- [general text to image](#general-text-to-image)
- [scene graph/layout to image](#scene-graphlayout-to-image)
- [dialog to image](#dialog-to-image)



## general text to image

| Note | Model       | Paper                                                        | Conference | paper link                                         | code link                                                    |
| ---- | ----------- | ------------------------------------------------------------ | ---------- | -------------------------------------------------- | ------------------------------------------------------------ |
|      | GAN-INT-CLS | Generative Adversarial Text to Image Synthesis               | ICML 2016  | [1605.05396](https://arxiv.org/abs/1605.05396)     | [reedscot/icml2016](https://github.com/reedscot/icml2016)    |
|      | StackGAN    | StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks | ICCV 2017  | [1612.03242](https://arxiv.org/abs/1612.03242)     | [hanzhanggit/StackGAN](https://github.com/hanzhanggit/StackGAN) |
|      | StackGAN++  | StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks | TPAMI 2018 | [1710.10916](https://arxiv.org/abs/1710.10916)     | [hanzhanggit/StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2) |
|      | AttnGAN     | AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks | CVPR 2018  | [1711.10485](https://arxiv.org/abs/1711.10485)     | [taoxugit/AttnGAN](https://github.com/taoxugit/AttnGAN)      |
|      | HD-GAN      | Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network | CVPR 2018  | [1802.09178](https://arxiv.org/pdf/1802.09178.pdf) | [ypxie/HDGan](https://github.com/ypxie/HDGan)                |
|      | StoryGAN    | StoryGAN: A Sequential Conditional GAN for Story Visualization | CVPR 2019  | [1812.02784](https://arxiv.org/abs/1812.02784)     | [yitong91/StoryGAN](https://github.com/yitong91/StoryGAN)    |
|      | MirrorGAN   | MirrorGAN: Learning Text-to-image Generation by Redescription | CVPR 2019  | [1903.05854](https://arxiv.org/abs/1903.05854)     |                                                              |
|      | DM-GAN      | DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis | CVPR 2019  | [1904.01310](https://arxiv.org/abs/1904.01310)     |                                                              |
|      | SD-GAN      | Semantics Disentangling for Text-to-Image Generation         | CVPR 2019  | [1904.01480](https://arxiv.org/abs/1904.01480)     |                                                              |

## scene graph/layout to image

| Note | Model           | Paper                                                        | Conference | paper link                                               | code link                                                    |
| ---- | --------------- | ------------------------------------------------------------ | ---------- | -------------------------------------------------------- | ------------------------------------------------------------ |
|      | GAWWN           | Learning What and Where to Draw                              | NIPS 2016  | [1610.02454](https://arxiv.org/abs/1610.02454)           | [reedscot/nips2016](https://github.com/reedscot/nips2016)    |
|      |                 | Inferring Semantic Layout for Hierarchical Text-to-Image Synthesis | CVPR 2018  | [1801.05091](https://arxiv.org/abs/1801.05091)           |                                                              |
|      | sg2im           | Image Generation from Scene Graphs                           | CVPR 2018  | [1804.01622](https://arxiv.org/abs/1804.01622)           | [google/sg2im](https://github.com/google/sg2im)              |
|      | Text2Scene      | Text2Scene: Generating Abstract Scenes from Textual Descriptions | CVPR 2019  | [1809.01110](https://arxiv.org/abs/1809.01110)           | [uvavision/Text2Image](https://github.com/uvavision/Text2Image) |
|      | Layout2Im       | Image Generation from Layout                                 | CVPR 2019  | [1811.11389](https://arxiv.org/abs/1811.11389)           |                                                              |
|      | LayoutGAN       | LayoutGAN: Generating Graphic Layouts with Wireframe Discriminator | ICLR 2019  | [openreview](https://openreview.net/forum?id=HJxB5sRcFQ) |                                                              |
|      | Object Pathways | Generating Multiple Objects at Spatially Distinct Locations  | ICLR 2019  | [1901.00686](https://arxiv.org/abs/1901.00686)           | [tohinz/multiple-objects-gan](https://github.com/tohinz/multiple-objects-gan) |
|      |                 | Using Scene Graph Context to Improve Image Generation        |            | [1901.03762](https://arxiv.org/abs/1901.03762)           |                                                              |
|      | Obj-GAN         | Object-driven Text-to-Image Synthesis via Adversarial Training | CVPR 2019  | [1902.10740](https://arxiv.org/abs/1902.10740)           |                                                              |

## dialog to image

| Note | Model       | Paper                                                        | Conference | paper link                                     | code link                                                    |
| ---- | ----------- | ------------------------------------------------------------ | ---------- | ---------------------------------------------- | ------------------------------------------------------------ |
|      | CoDraw      | CoDraw: Visual Dialog for Collaborative Drawing              |            | [1712.05558](https://arxiv.org/abs/1712.05558) | [CoDraw dataset](https://github.com/facebookresearch/CoDraw) |
|      | ChatPainter | ChatPainter: Improving Text to Image Generation using Dialogue | ICLRW 2018 | [1802.08216](https://arxiv.org/abs/1802.08216) |                                                              |
|      |             | Keep Drawing It: Iterative language-based image generation and editing | NIPSW 2018 | [1811.09845](https://arxiv.org/abs/1811.09845) | [CLEVR dataset](https://github.com/facebookresearch/clevr-dataset-gen) |
|      | Chat-crowd  | Chat-crowd: A Dialog-based Platform for Visual Layout Composition |            | [1812.04081](https://arxiv.org/abs/1812.04081) | [uvavision/chat-crowd](https://github.com/uvavision/chat-crowd) |
|      | SeqAttnGAN  | Sequential Attention GAN for Interactive Image Editing via Dialogue |            | [1812.08352](https://arxiv.org/abs/1812.08352) |                                                              |

