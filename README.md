# vit_b_32-model-fine-tuned-on-CIFAR-10
数据集在文件里，如果你想更换数据集，对ViT模型微调。首先下载你的数据集并解压，在代码中修改路径。其次根据数据集大小和你的意愿，选择合适的模型权重文件。之后在代码中reshape()你的图像形状适应模型权重文件，同时修改分类头到划分具体几类。 最后图像的数据集太大，则有必要采用GPU进行训练。

代码中图像数据集的下载地址：给出数据集下载网址：https://www.kaggle.com/c/cifar-10

ViT预训练模型权重下载链接：/kaggle/input/vit-base-models-pretrained-pytorch
