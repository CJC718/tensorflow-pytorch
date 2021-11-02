# tensorflow-pytorch

您好，我叫陈嘉诚，本科毕业太原理工大学通信工程专业，目前研二在读于东南大学人工智能应用专业，导师属于信息科学与工程学院。我目前研究方向是计算机视觉。研一我们的课程是全英文授课，英语可进行日常交流和文献阅读。目前学习了机器学习和深度学习相关知识。学习过的论文主要是，vgg，resnet，yolo系列和inception系列论文。也做了一些项目实现，比如基于词袋模型的数据集分类，使用yolov5模型实现对到道路区域物体的检测。目前求职意向是计算机视觉实习生岗位，我实习时间可以一年，可随时到岗，这就是我简单自我介绍，谢谢。

## 第一个项目 ##

提出了基于词袋模型的数据集分类解决方案，该方案是通过建立视觉词典对数据进行分类。
Bag-of-words模型是信息检索领域常用的文档表示方法。但是这里我们将Bag-of-words模型应用于图像表示。我们将图像看作文档，即若干个“视觉词汇”的集合。由于图像中的词汇不像文本中的那样是现成的，我们需要首先从图像中提取出相互独立的视觉词汇，这通常由三步组成。第一步，利用Hog算法提取图像的特征。第二步，使用Kmeans聚类算法将提取出的特征聚成k个簇，簇类特征具有高度相似性，这k个簇类的中心就是k个视觉词汇，这些视觉词汇构成视觉词典。第三步，利用视觉词典的中词汇表示图像。通过统计单词表中每个单词在图像中出现的次数，可以将图像表示成为一个K维数值向量。最后将测试集形成的频率直方图与训练集形成的频率直方图使用 KNN 进 直方图匹配，得到测试结果

## 第二个项目 ##

目的就是使用yolov5模型实现对到道路区域的物体的检测。首先，我进行了一些调研，因为yolov系列对设备比较友好，所以我就选取了yolov5来作为网络模型。然后我训练集选用coco数据集的特定数据类别的部分数据，比如car，person，bus等。这些在实际区域中经常出现。然后我使用make sense，手动标记了一些图片添加进数据集。第一次实验，我没有使用已经预训练好的网络，从头开始训练，训练大概四五个小时，训练结果很差，测试很多图片没有检测框。由于设备限制，我就选取一个已经训练好的最轻量化的网络yolo5s来训练，然后调整image-weights，multi-scale，label-smoothing，使得训练的模 型更加准确，泛化性更强。训练结束后，选取合适的目标检测阈值以及 IOU 阈值，得到最终预测结果。
结果:在经过三个多小时训练，300 次迭代之后，precision达到0.64，recall达到0.93，mAP@0.5达到0.91，mAP@0.5:0.95达到0.73。

## YoloV5 ##

 ![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%201.jpg)
 
 ![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%202.jpg)
 
 （1）输入端：Mosaic数据增强、自适应锚框计算、自适应图片缩放
（2）Backbone：Focus结构，CSP结构
（3）Neck：FPN+PAN结构
（4）Prediction：GIOU_Loss

Focus结构，在Yolov3&Yolov4中并没有这个结构，其中比较关键是切片操作。

比如右图的切片示意图，4*4*3的图像切片后变成2*2*12的特征图。

以Yolov5s的结构为例，原始608*608*3的图像输入Focus结构，采用切片操作，先变成304*304*12的特征图，再经过一次32个卷积核的卷积操作，最终变成304*304*32的特征图。

需要注意的是：Yolov5s的Focus结构最后使用了32个卷积核，而其他三种结构，使用的数量有所增加，先注意下，后面会讲解到四种结构的不同点。

## YoloV4 ##
![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%204.png)
先整理下Yolov4的五个基本组件：
CBM：Yolov4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成。
CBL：由Conv+Bn+Leaky_relu激活函数三者组成。
Res unit：借鉴Resnet网络中的残差结构，让网络可以构建的更深。
CSPX：借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concate组成。
SPP：采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。
Backbone：72层
输入端：这里指的创新主要是训练时对输入端的改进，主要包括Mosaic数据增强、cmBN、SAT自对抗训练
Mosaic数据增强则采用了4张图片，随机缩放、随机裁剪、随机排布的方式进行拼接。
为了解决COCO数据集中小目标分布不均的问题
丰富数据集：随机使用4张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好。
减少GPU：可能会有人说，随机缩放，普通的数据增强也可以做，但作者考虑到很多人可能只有一个GPU，因此Mosaic增强训练时，可以直接计算4张图片的数据，使得Mini-batch大小并不需要很大，一个GPU就可以达到比较好的效果。
SAT自对抗训练:第一个阶段中，神经网络更改原始图像；第二阶段中，训练神经网络以正常方式在修改后的图像上执行目标检测任务。

BackBone主干网络：将各种新的方式结合起来，包括：CSPDarknet53、Mish激活函数、Dropblock

Mish激活函数：
![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%205.png)

SPDarknet53是在Yolov3主干网络Darknet53的基础上，借鉴2019年CSPNet的经验，产生的Backbone结构，其中包含了5个CSP模块。
CSPNet的作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的。
因此采用CSP模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率。
因此Yolov4在主干网络Backbone采用CSPDarknet53网络结构，主要有三个方面的优点：
优点一：增强CNN的学习能力，使得在轻量化的同时保持准确性。
优点二：降低计算瓶颈
优点三：降低内存成本

Dropblock的研究者则干脆整个局部区域进行删减丢弃。
这种方式其实是借鉴2017年的cutout数据增强的方式，cutout是将输入图像的部分区域清零，而Dropblock则是将Cutout应用到每一个特征图。而且并不是用固定的归零比率，而是在训练时以一个小的比率开始，随着训练过程线性的增加这个比率。

Neck：目标检测网络在BackBone和最后的输出层之间往往会插入一些层，比如Yolov4中的SPP模块、FPN+PAN结构

SPP模块：自适应窗口大小，窗口的大小和activation map成比例，保证了经过pooling后出来的feature的长度是一致的.金字塔池化层有如下的三个优点，第一：他可以解决输入图片大小不一造成的缺陷。第二：由于把一个feature map从不同的角度进行特征提取，再聚合的特点，显示了算法的robust的特性。第三：同时也在object recongtion增加了精度。其实，你也可以这样想，最牛掰的地方是因为在卷积层的后面对每一张图片都进行了多方面的特征提取，他就可以提高任务的精度。

FPN+PAN结构：和Yolov3的FPN层不同，Yolov4在FPN层的后面还添加了一个自底向上的特征金字塔。其中包含两个PAN结构。这样结合操作，FPN层自顶向下传达强语义特征，而特征金字塔则自底向上传达强定位特征，两两联手，从不同的主干层对不同的检测层进行参数聚合,这样的操作确实很皮。
FPN+PAN借鉴的是18年CVPR的PANet，当时主要应用于图像分割领域，但Alexey将其拆分应用到Yolov4中，进一步提高特征提取的能力。

Prediction：输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数CIOU_Loss，以及预测框筛选的nms变为DIOU_nms
1.Iou 损失函数：1-Iou：状态1的情况，当预测框和目标框不相交时，IOU=0，无法反应两个框距离的远近，此时损失函数不可导，IOU_Loss无法优化两个框不相交的情况。
2. GIOU_Loss损失函数：最小外接矩形：增加了相交尺度的衡量方式，缓解了单纯IOU_Loss时的尴尬 

![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%206.jpg)

好的目标框回归函数应该考虑三个重要几何因素：重叠面积、中心点距离，长宽比

3. DIOU_Loss考虑了重叠面积和中心点距离，当目标框包裹预测框的时候，直接度量2个框的距离，因此DIOU_Loss收敛的更快。

![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%207.jpg) 

4. CIOU_Loss和DIOU_Loss前面的公式都是一样的，不过在此基础上还增加了一个影响因子，将预测框和目标框的长宽比都考虑了进去。

![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%208.png)  




## YoloV3 ##
Yolov3，网络结构主要由三个基本组件构成

![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%203.png) 

Concat：张量拼接，会扩充两个张量的维度，例如26*26*256和26*26*512两个张量拼接，结果是26*26*768。
整个yolo_v3_body包含252层
整个v3结构里面，是没有池化层和全连接层的。前向传播过程中，张量的尺寸变换是通过改变卷积核的步长来实现的。
yolo v3输出了3个不同尺度的feature map，如上图所示的y1, y2, y3。这也是v3论文中提到的为数不多的改进点：predictions across scales
这个借鉴了FPN(feature pyramid networks)，采用多尺度来对不同size的目标进行检测，越精细的grid cell就可以检测出越精细的物体。

v3对b-box进行预测的时候，采用了logistic regression。这一波操作sao得就像RPN中的线性回归调整b-box。v3每次对b-box进行predict时，输出和v2一样都是(tx,ty,th,tw,to)

v3的lossfunction由(x,y),(w,h),class,confidence组成。除了(w,h)，其他都采用交叉熵损失函数。

![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%209.png) 

## YoloV2 ##

BatchNormlization:

![image](https://user-images.githubusercontent.com/68943201/139191068-ba4c8d0c-c601-40db-90a9-6dde2c4c7f6c.png)

High Resolution Classifier:

对于YOLOv2，作者一开始在协调分类网络(指DarkNet-19)用的448X448全分辨率在ImageNet上跑了10个epoch。这使得网络有时间去调整自己的filter来使得自己能够在更高分辨率的输入上表现更佳。

Convolutional With Anchor Boxes: 在yolo_v2的优化尝试中加入了anchor机制。YOLO通过全连接层直接预测Bounding Box的坐标值。 yolov2会给出先验框，而每个先验框长短不一。

Direct location prediction:

multi-scale training

Fine-Grained Features：调整后的yolo将在13x13的特征图上做检测任务。虽然这对大物体检测来说用不着这么细粒度的特征图，但这对小物体检测十分有帮助。

![image](https://user-images.githubusercontent.com/68943201/139201402-5fba19a6-5422-478a-ae3a-e2de633d5b9e.png)


## YoloV1 ##

![image](https://user-images.githubusercontent.com/68943201/139201726-e14aab14-7055-4a9a-ab0b-4dac4650ca8d.png)

![image](https://user-images.githubusercontent.com/68943201/139201761-65513020-809a-4ce5-836e-c2e4d19f4d74.png)


## inceptionV1 ##

![image](https://user-images.githubusercontent.com/68943201/139204205-a1f93355-1e51-411f-ba2c-1365679278a6.png)

提出 Inception 结构，人为构建稀疏连接，引入多尺度感受野和多尺度融合
使用 [公式] 卷积层进行降维，减少计算量
使用均值池化取代全连接层，大幅度减少参数数目和计算量，一定程度上引入了正则化，同时使得网络输入的尺寸可变

辅助分类器 auxiliary classifiers

当网络深度相对较大时，可能需要担忧反向传播的能力。但是实验发现，在相对浅层的中间层，生成的特征具有较高的辨识度。

因此我们考虑添加辅助分类器 (auxiliary classifiers)。我们期望用它们来增加辨识度，从而补充梯度，并增加额外的正则化。辅助分类器由小的 CNNs 组成，接在 Icneption 模块之后。

事实上在较低的层级上这样处理基本没作用，作者在后来的 inception v3 论文中做了澄清。

在训练过程中，辅助分类器（Auxiliary classifiers） 的 loss 将会添加到总的 loss 中，但是权重较小 (本文为 0.3)。而在推理阶段，不需要这些 auxiliary classifiers。

其中，auxiliary classifiers 的具体结构如下：

均值池化层，后接 [公式] ，步长为 3 的卷积层，对应输出分别为： [公式] 和 [公式] 。
输出通道为 128 的 [公式] 的卷积层，用于维度缩减和非线性引入
全连接层，1024 个神经元，后接一个 ReLU
一个 dropout 层， [公式]
全连接层，后接一个 softmax 作为分类器，预测 1000 类输出


## inceptionV2 ##

将大卷积核换成堆叠的小卷机核，更少参数量，计算量，更多的非线性变化。空间可分离卷积。label smoothing

在特征（feature）维度上的稀疏连接进行处理，也就是在通道的维度上进行处理。

Batch Normalization

这个算法太牛了，使得训练深度神经网络成为了可能。从一下几个方面来介绍。

为了解决什么问题提出的BN

训练深度神经网络时，作者提出一个问题，叫做“Internal Covariate Shift”。

这个问题是由于在训练过程中，网络参数变化所引起的。具体来说，对于一个神经网络，第n层的输入就是第n-1层的输出，在训练过程中，每训练一轮参数就会发生变化，对于一个网络相同的输入，但n-1层的输出却不一样，这就导致第n层的输入也不一样，这个问题就叫做“Internal Covariate Shift”。

为了解决这个问题，提出了BN。

BN的来源

白化操作--在传统机器学习中，对图像提取特征之前，都会对图像做白化操作，即对输入数据变换成0均值、单位方差的正态分布。
卷积神经网络的输入就是图像，白化操作可以加快收敛，对于深度网络，每个隐层的输出都是下一个隐层的输入，即每个隐层的输入都可以做白化操作。 
在训练中的每个mini-batch上做正则化：
![image](https://user-images.githubusercontent.com/68943201/139216290-3754baf7-491d-49a1-a9aa-4e3536afa2f4.png)

![image](https://user-images.githubusercontent.com/68943201/139216649-d05063d2-8387-44bf-adb0-04c7dfd3c9d3.png)


BN的本质

我的理解BN的主要作用就是：

加速网络训练
防止梯度消失
如果激活函数是sigmoid，对于每个神经元，可以把逐渐向非线性映射的两端饱和区靠拢的输入分布，强行拉回到0均值单位方差的标准正态分布，即激活函数的兴奋区，在sigmoid兴奋区梯度大，即加速网络训练，还防止了梯度消失。

基于此，BN对于sigmoid函数作用大。

sigmoid函数在区间[-1, 1]中，近似于线性函数。如果没有这个公式：

就会降低了模型的表达能力，使得网络近似于一个线性映射，因此加入了scale 和shift。

它们的主要作用就是找到一个线性和非线性的平衡点，既能享受非线性较强的表达能力，有可以避免非线性饱和导致网络收敛变慢问题。

## Xception ##

把输入的各个通道单独用一个卷积核，再使用1*1卷积把跨通道信息结合。处理不同信息的通道，

## Resnet ##

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图5所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。

Resnet 解决了网络退化现象
为什么可以解决网络退化：
深层梯度回传顺畅：恒等映射这一路梯度为1，把深层梯度注入到底层，防止梯度消失。没有中间层层层盘剥。
残差分支修正上一层的误差；
传统线性结构网络很难拟合恒等映射；
有时候原始信息也很重要。可以让模型自行选择要不要更新。弥补了高度非线性造成的不可逆的信息损失。
ResNet反向传播传回的梯度相关性好。
Resnet 相当于几个浅层网络的集成，有很多潜在路径
Skip connection可以实现不同分辨率特征的组合

## Faster rcnn ##

Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

上图4展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

## Fast- rcnn ##

RCCN存在以下几个问题：
（1） 需要事先提取多个候选区域对应的图像。这一行为会占用大量的磁盘空间；
（2） 针对传统的CNN来说，输入的map需要时固定尺寸的，而归一化过程中对图片产生的形变会导致图片大小改变，这对CNN的特征提取有致命的坏处；
（3） 每个region proposal都需要进入CNN网络计算。进而会导致过多次的重复的相同的特征提取，这一举动会导致大大的计算浪费。
　　针对以上问题，Fast R-CNN采用了几个更新来提高训练和测试速度，同时也提高了检测精度。在本文中，简化了基于最先进的卷积神经网络的对象检测器的训练过程。提出了一种单阶段联合学习的目标建议分类和空间定位的训练算法。

　ROI是指的在SS完成后得到的“候选框”在卷积特征图上的映射；将每个候选区域均匀分成H×W块，对每块进行max pooling。将特征图上大小不一的候选区域转变为大小统一的数据，送入下一层。如下图：
 
  这样虽然输入的图片尺寸不同，得到的feature map（特征图）尺寸也不同，但是可以加入这个神奇的ROI Pooling层，对每个region都提取一个固定维度的特征表示，就可再通过正常的softmax进行类型识别（在论文中使用VGG16，故需提取为7×7）。每个RoI由一个四元组(r, c, h, w)定义，该四元组指定其左上角(r, c)及其高度和宽度(h, w)。
　　以上操作避免了RCNN存在让图像产生形变，或者图像变得过小的问题，使一些特征产生了损失，继而对之后的特征选择产生巨大影响。

