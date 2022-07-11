"""
@Author: Conghao Wong
@Date: 2021-04-30 15:09:20
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:07:34
@Description: Basic transformer structure. Part of modules come 
              from https://www.tensorflow.org/tutorials/text/transformer.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf


def get_angles(pos: np.ndarray, i: np.ndarray, d_model:int):
    """
    Get the relative position representation in angles.

    :param pos: position of all inputs
    :param i: locations in d-dimension space
    :param d_model: number of all inputs
    :return angles: angles of all inputs
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    ## 位置编码（Positional encoding）

    因为该模型并不包括任何的循环（recurrence）或卷积，
    所以模型添加了位置编码，为模型提供一些关于单词在句子中相对位置的信息。

    位置编码向量被加到嵌入（embedding）向量中。
    嵌入表示一个 d 维空间的标记，在 d 维空间中有着相似含义的标记会离彼此更近。
    但是，嵌入并没有对在一句话中的词的相对位置进行编码。
    因此，当加上位置编码后，词将基于*它们含义的相似度以及它们在句子中的位置*，
    在 d 维空间中离彼此更近。

    参看 [位置编码](https://github.com/tensorflow/examples/blob/master/community/en/position_encoding.ipynb) 的 notebook 了解更多信息。计算位置编码的公式如下：

    $$\Large{PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})} $$
    $$\Large{PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})} $$
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """    
    遮挡一批序列中所有的填充标记（pad tokens）。
    这确保了模型不会将填充作为输入。
    该 mask 表明填充值 `0` 出现的位置：在这些位置 mask 输出 `1`，否则输出 `0`。
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    前瞻遮挡（look-ahead mask）用于遮挡一个序列中的后续标记（future tokens）。
    换句话说，该 mask 表明了不应该使用的条目。

    这意味着要预测第三个词，将仅使用第一个和第二个词。
    与此类似，预测第四个词，仅使用第一个，第二个和第三个词，依此类推。 
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_encoder_mask(inp):
    return create_padding_mask(inp)[:, :, :, :, 0]


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)[:, :, :, :, 0]
    
    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)[:, :, :, :, 0]
    
    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)[:, :, :, :, 0]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    <img src="https://tensorflow.google.cn/images/tutorials/transformer/scaled_attention.png" width="500" alt="scaled_dot_product_attention">

    Transformer 使用的注意力函数有三个输入：Q（请求（query））、K（主键（key））、V（数值（value））。
    用于计算注意力权重的等式为：

    $$\Large{Attention(Q, K, V) = softmax_k(\frac{QK^T}{\sqrt{d_k}}) V} $$

    点积注意力被缩小了深度的平方根倍。
    这样做是因为对于较大的深度值，点积的大小会增大，从而推动 softmax 函数往仅有
    很小的梯度的方向靠拢，导致了一种很硬的（hard）softmax。

    例如，假设 `Q` 和 `K` 的均值为0，方差为1。
    它们的矩阵乘积将有均值为0，方差为 `dk`。
    因此，*`dk` 的平方根*被用于缩放（而非其他数值），
    因为，`Q` 和 `K` 的矩阵乘积的均值本应该为 0，方差本应该为1，
    这样会获得一个更平缓的 softmax。

    遮挡（mask）与 -1e9（接近于负无穷）相乘。
    这样做是因为遮挡与缩放的 Q 和 K 的矩阵乘积相加，
    并在 softmax 之前立即应用。
    目标是将这些单元归零，因为 softmax 的较大负数输入在输出中接近于零。

    计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。
    
    参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。
        
    返回值:
        输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    <img src="https://tensorflow.google.cn/images/tutorials/transformer/multi_head_attention.png" width="500" alt="multi-head attention">


    多头注意力由四部分组成：
    *    线性层并分拆成多头。
    *    按比缩放的点积注意力。
    *    多头及联。
    *    最后一层线性层。

    每个多头注意力块有三个输入：Q（请求）、K（主键）、V（数值）。
    这些输入经过线性（Dense）层，并分拆成多头。 

    将上面定义的 `scaled_dot_product_attention` 函数应用于每个头
    （进行了广播（broadcasted）以提高效率）。
    注意力这步必须使用一个恰当的 mask。
    然后将每个头的注意力输出连接起来（用`tf.transpose` 和 `tf.reshape`），
    并放入最后的 `Dense` 层。

    Q、K、和 V 被拆分到了多个头，而非单个的注意力头，
    因为多头允许模型共同注意来自不同表示空间的不同位置的信息。
    在分拆后，每个头部的维度减少，因此总的计算成本与有着全部维度的单个注意力头相同。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    ## 点式前馈网络（Point wise feed forward network）

    点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  
        # (batch_size, seq_len, dff)

        tf.keras.layers.Dense(d_model)  
        # (batch_size, seq_len, d_model)
    ])