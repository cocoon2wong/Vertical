"""
@Author: Conghao Wong
@Date: 2021-04-30 14:58:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 15:07:15
@Description: Basic transformer structure. Part of modules come 
              from https://www.tensorflow.org/tutorials/text/transformer.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ._utils import (MultiHeadAttention, create_encoder_mask, create_masks,
                     point_wise_feed_forward_network, positional_encoding)


class EncoderLayer(tf.keras.layers.Layer):
    """
    ### 编码器层（Encoder layer）

    每个编码器层包括以下子层：

    1.   多头注意力（有填充遮挡）
    2.   点式前馈网络（Point wise feed forward networks）。

    每个子层在其周围有一个残差连接，然后进行层归一化。
    残差连接有助于避免深度网络中的梯度消失问题。

    每个子层的输出是 `LayerNorm(x + Sublayer(x))`。
    归一化是在 `d_model`（最后一个）维度完成的。
    Transformer 中有 N 个编码器层。
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    ### 解码器层（Decoder layer）

    每个解码器层包括以下子层：

    1.   遮挡的多头注意力（前瞻遮挡和填充遮挡）
    2.   多头注意力（用填充遮挡）。
            V（数值）和 K（主键）接收*编码器输出*作为输入。
            Q（请求）接收*遮挡的多头注意力子层的输出*。
    3.   点式前馈网络

    每个子层在其周围有一个残差连接，然后进行层归一化。
    每个子层的输出是 `LayerNorm(x + Sublayer(x))`。
    归一化是在 `d_model`（最后一个）维度完成的。

    Transformer 中共有 N 个解码器层。

    当 Q 接收到解码器的第一个注意力块的输出，并且 K 接收到编码器的输出时，
    注意力权重表示根据编码器的输出赋予解码器输入的重要性。
    换一种说法，解码器通过查看编码器输出和对其自身输出的自注意力，预测下一个词。
    参看按比缩放的点积注意力部分的演示。
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """
    This module comes from https://www.tensorflow.org/tutorials/text/transformer.

    ### 编码器（Encoder）

    `编码器` 包括：
    1.   输入嵌入（Input Embedding）
    2.   位置编码（Positional Encoding）
    3.   N 个编码器层（encoder layers）

    输入经过嵌入（embedding）后，该嵌入与位置编码相加。
    该加法结果的输出是编码器层的输入。
    编码器的输出是解码器的输入。
    """

    def __init__(self, num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 rate=0.1):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # original transformer for translation
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # transformer for trajectory prediction
        self.embedding = tf.keras.layers.Dense(d_model, activation=tf.nn.tanh)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """
    ### 解码器（Decoder）

    `解码器`包括：
    1.   输出嵌入（Output Embedding）
    2.   位置编码（Positional Encoding）
    3.   N 个解码器层（decoder layers）

    目标（target）经过一个嵌入后，该嵌入和位置编码相加。
    该加法结果是解码器层的输入。
    解码器的输出是最后的线性层的输入。
    """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # original transformer for translation
        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        # transformer for trajectory prediction
        self.embedding = tf.keras.layers.Dense(d_model, activation=tf.nn.tanh)

        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    ## 创建 Transformer

    Transformer 包括编码器，解码器和最后的线性层。
    解码器的输出是线性层的输入，返回线性层的输出。
    """

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,
                 pe_target,
                 rate=0.1,
                 include_top=True,
                 training_structure=None,
                 *args, **kwargs):
        """
        Init a transformer

        :param num_layers: number of encoder/decoder layers
        :param d_model: feature dimensions
        :param num_heads: 
            number of head in multi-head attention layers.
            Note that `d_model % num_heads == 0`
        :param dff: dimension of feed forward networks
        :param input_vocab_size:
            ***USELESS for trajectory prediction***
            vocabulary's number of embadding layer when encoding
        :param target_vocab_size:
            ***set to `2` for trajectory prediction***
            vocabulary's number of embadding layer when decoding
        :param pe_input: maximum position when encoding positions in encoder
        :param pe_target: maximum position when encoding positions in decoder
        :param rate: dropout rate
        """

        super().__init__()

        self.include_top = include_top

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        if self.include_top:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs: tf.Tensor,
             targets: tf.Tensor,
             training=None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Transformer forward implementation

        :param inputs: inputs tensor to the transformer encoder, shape = `(batch, M, N)`
        :param targets: inputs tensor to the transformer decoder, shape = `(batch, A, B)`
        :param training: controls if training or test

        :return outputs: transformer outputs tensor
        :return attention_weights: attention weights
        """

        # create masks
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(
            inputs, targets
        )

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inputs, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(targets,
                                                     enc_output,
                                                     training,
                                                     look_ahead_mask,
                                                     dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(
            dec_output) if self.include_top else dec_output

        return final_output, attention_weights


class TransformerEncoder(tf.keras.Model):
    """
    Transformer Encoder
    """

    def __init__(self, num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 dim_forward: int,
                 steps: int,
                 dim_input: int = None,
                 dim_output: int = None,
                 dropout=0.1,
                 include_top=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.include_top = include_top

        # Transformer Encoder
        self.TE = Encoder(num_layers=num_layers,
                          d_model=dim_model,
                          num_heads=num_heads,
                          dff=dim_forward,
                          input_vocab_size=dim_input,
                          maximum_position_encoding=steps,
                          rate=dropout)

        if self.include_top:
            self.final_layer = tf.keras.layers.Dense(dim_output)

    def call(self, inputs: tf.Tensor, training=None, *args, **kwargs) -> tf.Tensor:

        # Create masks
        enc_mask = create_encoder_mask(inputs)

        # Transformer Encoder -> (batch, steps, dim_input)
        output = self.TE(inputs, training, enc_mask)

        # Top layer -> (batch, steps, dim_output)
        if self.include_top:
            output = self.final_layer(output)

        return output
