# Learning Resolution-Adaptive Representations for Cross-Resolution Person Re-Identification

IEEE TRANSACTIONS ON IMAGE PROCESSING

- varying-length representation
- resolution-adaptive mechanism
- We propose a novel progressive training strategy

In contrast, this paper proposes a method that has its own differences: first, our method does not need excessive feature extraction for cross-resolution matching; and second we directly learn resolution-adaptive representations which are amenable for cross-resolution comparison.

![f2](./f2.png)

## PROPOSED METHOD

### Problem Statement and Framework Overview

训练的目标是学习resolution-adaptive 度量，如下

$$dist(x_p, x_g) = M(x_p, x_g, k)$$

其中M是训练的模型，x_p是query，x_g是gallery，k 表示分辨率比例——与最高的分辨率的长/宽的比例e.g., k ∈ {1, 1/2, 1/3, 1/4}；
k使得此度量是resolution-adaptive

For example, if the highest resolution corresponds to 256 × 128 per person image, i.e., its resolution ratio k = 1, for a LR image of size 64×32, its resolution ratio becomes k = 1/4.

In our algorithm, we resize all the images, whether LR or HR,to equal to the size of the highest resolution images via bilinear up-sampling. Then, each full-resolution image is down-sampled at different specified ratios to form its LR alternations.

>训练中使用的LR也是人工生成的——HR减采样再双线性插值——将所有图像resize到同一大小，也许是方便训练

假设分辨率k是可预测的；

We assume that the resolutions of both the query and gallery images are provided. In practice, the resolution could be estimated from the size (number of pixels) of images or pedestrian bounding boxes since the height of people is relatively fixed

关于M的实现，M学习resolution-adaptive表示，通过一下两个机制实现:

- varying-length representation
  - varying dimensions to encode a query image
- injecting resolutionspecific masks
  - inject into the intermediate residual feature blocks
  - to further extract resolution-specific information

resolution-adaptive representations与resolution-invariant features对比：`However, since the resolution of the query image is not fixed, learning resolution-invariant features will identify discriminative information that are shared across all resolutions. Consequently, the information specific to resolutions higher than the lowest one will not be preserved.This inevitably prevent the network from using more information for matching a moderate LR query to HR gallery images.`

### Mechanism 1: Learning Varying-Length Resolution-Adaptive Representations

![f1](image-14.png)

**motivation:**`a HR image should contain all the information conveyed in the LR image, but also extra information from the higher resolution`

When one compares a HR image and a LR image, the comparison should only be based on the shared part

In CRReID, a query image could have different resolutions, thus the above strategy will result in different representation lengths, i.e., the higher resolution of the query is, the more information that can be shared with the HR gallery images, and thus the longer dimension of the representation is.

In our implementation, we define m sub-vectors $\{\mathbf{v}_k\}, k = 1, \cdots , m$

> 这里的k与之前的分辨率比例k不一样；这里的k表示分辨率水平，k越大对应的分辨率越高

Varying-Length Resolution-Adaptive Representations 用下面的$\mathbf{z}$表示

$$
\begin{align*}
\mathbf{z}_p&=cat(\mathbf{v}_1^p,\cdots,\mathbf{v}_k^p,\mathbf{v}_{k+1:m}^p)\\
\mathbf{z}_g&=cat(\mathbf{v}_1^g,\cdots,\mathbf{v}_k^g,\cdots,\mathbf{v}_{m}^g)\\
\mathbf{\hat{z}}_p&=cat(\mathbf{v}_{1:k}^p)\\
\mathbf{\hat{z}}_g&=cat(\mathbf{v}_{1:k}^g)\\
dis(x_p,x_g)&=\lVert \mathbf{z}_p-\mathbf{z}_g\rVert _2^2=\lVert \mathbf{\hat{z}}_p-\mathbf{\hat{z}}_g\rVert _2^2
\end{align*}
$$

其中，$\mathbf{z}_p$后面的$\mathbf{v}_{k+1:m}^p$是0；p表示query，g表示gallery

>后面的分类似乎不是直接这样用？或许这里的距离并不是欧氏距离

### Mechanism 2: Resolution-Adaptive Masking

The above varying-length representation only adaptively constructs the resolution-specific representation in the **penultimate layer** of the neural network. To extract more resolutiondependent features, we propose a mechanism to inject the resolution characteristics into the earlier layers of a neural network.

> 在最后得到Resolution-Adaptive Representations时使用到query分辨率信息，得到representation后还有一步分类，所以是倒数第二层

文章使用残差神经网络，往每一层残差块输出插入mask，mask集合定义为$\{\mathbf{M}_k^l\in \mathbb{R}^{d^l}\},k=1,\cdots,m$；k表示分辨率水平，l表示第l个残差块

Each mask is a vector, with each dimension being a real value between 0 and 1.The mask acts as a dimension-wise scaling factor to the feature maps

按以下公式应用mask
$$
\mathbf{\bar{X}}^l=\mathbf{X}^l \odot \mathbf{M}_k^l\\
$$
其中$\odot$表示逐元素乘法
In practice:
$$
\begin{align*}
  &\mathbf{\bar{X}}^l=\mathbf{X}^l \odot (\sum _k s_k^l\mathrm{Sigmoid}(\mathbf{M}_k^l))\\
  &s_k^l=
  \begin{cases}
    1,&\text{if the input image is at resolution level k}\\
    0,&\text{others}
  \end{cases}\\
  &\mathrm{Sigmoid}(x)=\frac{1}{1+e^{-x}}
\end{align*}
$$
实际使用上，$\mathbf{M}_k^l$可能不是0，1之间的，用Sigmoid转换为0，1之间。
实际使用中，把所有mask放在矩阵中，每列表示分辨率水平k，行表示第l个残差块
对通道做掩模的解释：`We recall that developing mask generators is equivalent to aligning person images with occlusion, wherein visible patterns from non-occluded images can be selected by corresponding masks to align and compare with occluded regions [3], [4]. It is worth noting that our proposed resolution-dependent masks are applied in a channel-wise manner to reflect the resolution levels in feature dimensions, making them suitable for CRReID`

### Varying-Length Sub-Vectors With Resolution Variations

Since the LR image shares content with the original HR image but also contains its own characteristics, the feature vector of each image should be a combination of commonality and resolution-induced characteristics.

However, a deep feature representation outputted from neural networks has a fixed-size dimension, making it challenging to define varied feature dimension corresponding to different resolution levels.

> representation **z**是多个子向量的拼接，是变长的，与分辨率水平相关，而神经网络输出的是固定维数的

To overcome this challenge, we propose to predict a set of sub-vectors, where the number of sub-vectors corresponds to the resolution.
Specifically, we train a classifier consisting of a set of sub-classifiers, such that an image at a resolution looks up the sub-classifiers to adaptively characterise its own features

representaion **v**(或者是之前的**z**)用一系列子向量表示，也要训练包含一系列子分类器的分类器**W**，用**e**进行分类预测

$$
\begin{align*}
  \mathbf{v}&=cat(\mathbf{v}_1,\cdots ,\mathbf{v}_k,\cdots,\mathbf{v}_m)\in \mathbb{R}^d,~
  \mathbf{v}_k\in \mathbb{R}^{d_k}\\
  \mathbf{W}&=cat(\mathbf{w}^1,\cdots ,\mathbf{w}^k,\cdots,\mathbf{w}^C)\in \mathbb{R}^{d\times C},~
  \mathbf{w}^k\in \mathbb{R}^{d_k\times C}\\
  \mathbf{e}^k&=(\mathbf{w}^k)^T\mathbf{v}_k~\in \mathbb{R}^C
\end{align*}
$$

> 此处用**v**表示特征，应该是和之前的**z**是一个东西，而且这里改用cat更好，保持一致；文章后续**v**与**z**似乎混用了

Since the identity prediction classifies each image by evaluating the classifier w k into the embedding space, the classifier can be interpreted as the prototype closest to the image in the feature space.

> **e**实际上是向量**v,w**的点积，**v**越接近**w**，**e**越大，所以说**w**是在embedding空间中最接近图像的原型；具体**e**是怎么使用呢，似乎不是越大越接近图像
>**e**做softmax应该会得到那个分类结果的one-hot向量，就是分类结果

### Resolution-Adaptive Representation Training

训练模型使用的loss:

- identity loss $\mathcal{L}_{cls}$
  - softmax 之后再求 交叉熵(cross-entropy)
- verification loss $\mathcal{L}_{verif}$
  - applied to a binary classifier that predicts whether two samples belong to the same class

$$
\mathcal{L}_{verif}=-\sum _n^N[y_n\mathrm{log}(p(y_n=1|\mathbf{v}_{ij}))\\
+(1-y_n)\mathrm{log}(1-p(y_n=1|\mathbf{v}_{ij}))]\\
\mathbf{v}_{ij}=\mathbf{v}_{i}-\mathbf{v}_{j}\\
\mathcal{L}_{cls}+\lambda\mathcal{L}_{verif}
$$

实验中，$p(y_n=1|\mathbf{v}_{ij})=Sigmoid(f(\mathbf{v}_{ij}))$，是MLP的输出

> 文中提到MLP输出是 scale 而这里分类器的输出**e**应该是 onehot 向量，难道是再训练一个不同的分类器？

In our implementation, we start by padding zeros to the varying-length representations of two images, and send their feature vector difference to a multi-layer perceptron (MLP) to make a **binary prediction** about whether those two samples are from the same class

![loss](image-16.png)

#### Analysis of the Identity Classification Loss

> 这里解释为什么这两个loss可以有利于resolution-adaptive metric learning；没看懂为什么verification loss 可以 learn a resolution-adaptive metric naturally

The verification loss takes inputs from two images from different or same resolutions. It can learn a resolution-adaptive metric naturally

> 应该是因为MLP输入的是$\mathbf{v}_{ij}=\mathbf{v}_{i}-\mathbf{v}_{j}$(或者说是**z**)，直接相减有距离的概念和前面想要求的dist呼应？好像也不是，要呼应**变长、分辨率自适应**

we propose to apply zero-padding, i.e., concatenating “0” to the representation whose dimension is less than the maximal dimension, to convert a varying-length representation to a fixed-length representation.

The ID loss will encourage samples from the same identity class to move closer to the corresponding classifier w and thus indirectly pulling those features close to each other.Similarly, we could expect our ID loss will pull $\mathbf{z}_k$ and the first k-th sub-vectors of $\mathbf{z}_{k'},k' >k$ close to each other, which ensures that images of the same identity but different resolutions become closer under the proposed distance metric Eq.

> 对于ID loss，会对得到**z**进行补零使得长度一致；**W**与**z**做内积，补的零可以使得与分类结果与分辨率相关；各自分辨率中，**z**会靠近，不同分辨率的**z**也会靠近

#### Progressive Training for Resolution-Adaptive Masks

文章指出想要联合对两个机制进行优化，而传统的stochastic gradient descent(随机梯度下降)，因为对不共享掩码进行优化困难，会导致性能受损。此外，不同层的通道掩码是高度相关的，并且由于共适应问题，训练这些掩码变得不重要。

为了解决这个问题，我们提出了一个有效的渐进式培训计划。我们建议在不同的层顺序注入掩模，并逐步训练它们，以避免多个掩模的协同适应

在我们的实现中，我们首先将掩模制作成最靠近分类器层的残差块，然后将更多的掩模向下逐渐乘以残差块以降低。一旦新的掩码被编织到剩余块，用以前更高级别训练过的掩码将被固定，不再更新。

整个训练算法如下

![Algorithm1](image-15.png)

## IMPLEMENTATION DETAILS
