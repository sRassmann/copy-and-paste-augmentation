# Instance Segmentation on Large Scale Images

This is an university project of the [Lab "Intelligent Vision Systems"](http://vsteinhage.github.io/HTML/pg.html) during the summer term 2021 at the University of Bonn.

I'm adapting
[FAIRs Tensormask implemented in detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/TensorMask)
to be applied to large image stiches of scanned bug and butterfly collection boxes. In order to leverage sparse training data (7 annotated images with about 700 instances in 7 classes) I am implementing a [Copy & Paste Data Augmentation](https://arxiv.org/abs/2012.07177) to generate training images on-the-fly during training.



![Butterflies](http://vsteinhage.github.io/Images/Labs/Butterflies.jpg)
