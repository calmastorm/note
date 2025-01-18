# The Illustrated Transformer (Note)

> 这是我读*Jay Alammar*的*The Illustrated Transformer*时的一些笔记。本科时我囫囵吞枣般学习了一下Transformer，但并没有深入学习。但近期，因为教授们都提到Transformer非常前沿非常重要，而且朋友告诉我其在CV上的效果甚至会比Deep Learning要好（出乎我的意料），因此我认为读一下这篇文章并深入学习一下Transformer的技术要点是有必要的，希望还不算太晚。

本人不拥有文中任何图片，如果侵权了请通知我删除。

## Architecture

我们可以从一个最简单的翻译功能来着手学习。下面这个图可以看出Transformer整体可以看成一个大函数大机器，输入A语言的句子，输出一个B语言的句子。

![the_transformer_3](../img/the_transformer_3.png)

Transformer本体又由Encoders和Decoders组成。最初的输入会先进入Encoder，最终的输出从Decoder出来。Encoders们最终的输出又会给到Decoders们。

![The_transformer_encoders_decoders](../img/The_transformer_encoders_decoders.png)

再次拆分，Encoders由多个Encoder组成，而Decoders们有着同样的数量。这里可以看到最后一个Encoder的输出是要传递给每一个Decoder来使用的。

![The_transformer_encoder_decoder_stack](../img/The_transformer_encoder_decoder_stack.png)

### Encoder

每个Encoder的结构一模一样，但内部的权重不一样。每个Encoder由两个主要部分组成，分别是一个Self-Attention层和一个Feed Forward NN层。Self-Attention层让Encoder层在给某个输入编码时，同时看向其他的输入，后续有更详细的笔记。Feed Forward NN层则是一个MLP。

> 文中我并没有找到FFN的详细讲解，所以下面我对FFN的了解来自CSDN。FFN本质上是一个两层的MLP，数学本质是这样的：
>
> ![e839a2189be04a440a79b9f1f783dd80](../img/e839a2189be04a440a79b9f1f783dd80.png)
>
> 其中两层感知机中，第一层会将输入的向量升维，第二层将向量重新降维。这样子就可以学习到更加抽象的特征。（来自Odd Function的CSDN笔记）

![Transformer_encoder](../img/Transformer_encoder.png)

这里顺便提一句Decoder，每个Decoder的结构都与Encoder很相似，但每个Decoder都多了一个Encoder-Decoder Attention层在中间。这个EDA层让Decoder专注于被输入句子中的相关部分。举个例子，当一个*them*出现时，是指代句子中提到的*cats*还是*dogs*就看它了。

![Transformer_decoder](../img/Transformer_decoder.png)

### Tensors

Tensors就是Vectors，接下来是关于他们的作用，意义，以及他们是如何在整个Transformer里流转的。

[未完成]

## Links

[The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

[从零开始了解transformer的机制|第四章：FFN层的作用 by Odd Function](https://blog.csdn.net/weixin_73179708/article/details/132516512)

