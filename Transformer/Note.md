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

简单来说，我们会把每个输入的词，通过一个词嵌入算法（Embedding algorithm），让他们都变成向量。在原论文中，每个向量的长度都是512，下面图片为了演示使用了长度为4的向量。

![embeddings](../img/embeddings.png)

在最底层的encoder下面，才有这个词嵌入，其他的encoder的输入都是前一个encoder的输出。关于有多少个这样的向量，一般是最长的句子的尺寸。假如一个数据集中最长的句子有50个词，那么就有50个向量。

![encoder_with_tensors](../img/encoder_with_tensors.png)

## Encoding

这里再提一次Encoder的整体结构，一个Encoder接收一个列表的向量作为输入，通过把他们放入SA层进行处理，然后再进入FFNN层处理（注意其实只有一个FFNN），最后把这个输出传给下一个Encoder。

![encoder_with_tensors_2](../img/encoder_with_tensors_2.png)

### Self-Attention

Self-Attention的作用其实很简单，就是让每个词都能够通过观察其他词的位置来获取更多信息，让encoding效果更好。举个例子，<u>The animal didn't cross the street because it was too tired</u>，里面的it，人们都知道说的是The animal，电脑却不一定知道。因此，自注意力机制就帮我们更好地处理这个问题。

下面这个图片是最后一个encoder的Attention情况，可以看到it已经专注到The animal上面了。

![transformer_self-attention_visualization](../img/transformer_self-attention_visualization.png)

### Self-Attention in Details

要计算self-attention的第一步，从encoder的输入向量中，创建三个向量，也就是q、k、v。具体来说，对于每个词，我们都要有一个Query向量、Key向量、Value向量。这些向量是由输入的向量和WQ、WK、WV权重使用矩阵乘法得出的，看下面的图。注意这些新向量比词嵌入向量小，图片中展示的3格来表示长度为64，别忘了4格是512。

![transformer_self_attention_vectors](../img/transformer_self_attention_vectors.png)

先跳过QKV具体的意义，现在只需要知道这些对于计算attention有帮助就可以了。

计算self-attention的第二步就是计算一个score，这决定了每个词要对其他position有多少关注。比方说，我们已经通过第一步算出了所有词的qkv，我们就取q1和k1的点乘作为position1的第一个score，再取q1和k2的点乘作为position1第二个score。

![transformer_self_attention_score](../img/transformer_self_attention_score.png)

第三步和第四步结合来看，就是计算他们的softmax，首先要使用每个score除以k向量长度的开根方，别忘了k长度为64，它长度的开根方就是8。最后我们把得到的position1的score1和score2进行softmax计算，得出概率。

![self-attention_softmax](../img/self-attention_softmax.png)

第五步就是让softmax得出的概率直接乘以v。按照图中的例子，0.88先乘以v1，然后0.12乘以v2。这样的目的是让无关的词乘以一个很小的概率值，比如说0.001，确保他们被忽略。

第六步就是把得到的这些加权后的v向量们都加起来，得出z1，这就是position1这个词在self-attention层的输出了。其他的positions也是一样的计算。

![self-attention-output](../img/self-attention-output.png)

### Matrix Calculation

【未完成】

## Links

[The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

[从零开始了解transformer的机制|第四章：FFN层的作用 by Odd Function](https://blog.csdn.net/weixin_73179708/article/details/132516512)

