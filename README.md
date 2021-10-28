# tensorflow-pytorch
## 第一个项目 ##

提出了基于词袋模型的数据集分类解决方案，该方案是通过建立视觉词典对数据进行分类。
Bag-of-words模型是信息检索领域常用的文档表示方法。但是这里我们将Bag-of-words模型应用于图像表示。我们将图像看作文档，即若干个“视觉词汇”的集合。由于图像中的词汇不像文本中的那样是现成的，我们需要首先从图像中提取出相互独立的视觉词汇，这通常由三步组成。第一步，利用Hog算法提取图像的特征。第二步，使用Kmeans聚类算法将提取出的特征聚成k个簇，簇类特征具有高度相似性，这k个簇类的中心就是k个视觉词汇，这些视觉词汇构成视觉词典。第三步，利用视觉词典的中词汇表示图像。通过统计单词表中每个单词在图像中出现的次数，可以将图像表示成为一个K维数值向量。最后将测试集形成的频率直方图与训练集形成的频率直方图使用 KNN 进 直方图匹配，得到测试结果

## 第二个项目 ##

目的就是使用yolov5模型实现对到道路区域的物体的检测。首先，我进行了一些调研，因为yolov系列对设备比较友好，所以我就选取了yolov5来作为网络模型。然后我训练集选用coco数据集的特定数据类别的部分数据，比如car，person，bus等。这些在实际区域中经常出现。然后我使用make sense，手动标记了一些图片添加进数据集。第一次实验，我没有使用已经预训练好的网络，从头开始训练，训练大概四五个小时，训练结果很差，测试很多图片没有检测框。由于设备限制，我就选取一个已经训练好的最轻量化的网络yolo5s来训练，然后调整image-weights，multi-scale，label-smoothing，使得训练的模 型更加准确，泛化性更强。训练结束后，选取合适的目标检测阈值以及 IOU 阈值，得到最终预测结果。
结果:在经过三个多小时训练，300 次迭代之后，对行人检测置信度达到 85%以上，对广告牌以及标志牌的 置信度达到 60%，对手机等小物体置信度达到 20%。

## YoloV5 ##

 ![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%201.jpg)
 
 ![image](https://github.com/CJC718/tensorflow-pytorch/blob/main/githup%E5%9B%BE%E7%89%87/%E5%9B%BE%E7%89%87%202.jpg)

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


Concat：张量拼接，会扩充两个张量的维度，例如26*26*256和26*26*512两个张量拼接，结果是26*26*768。
整个yolo_v3_body包含252层
整个v3结构里面，是没有池化层和全连接层的。前向传播过程中，张量的尺寸变换是通过改变卷积核的步长来实现的。
yolo v3输出了3个不同尺度的feature map，如上图所示的y1, y2, y3。这也是v3论文中提到的为数不多的改进点：predictions across scales
这个借鉴了FPN(feature pyramid networks)，采用多尺度来对不同size的目标进行检测，越精细的grid cell就可以检测出越精细的物体。

v3对b-box进行预测的时候，采用了logistic regression。这一波操作sao得就像RPN中的线性回归调整b-box。v3每次对b-box进行predict时，输出和v2一样都是
## YoloV4 ##

