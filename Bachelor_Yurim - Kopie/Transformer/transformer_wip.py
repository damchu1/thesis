# The purpose of this Python script is to define the individual
# components constituting the state-of-the-art transformer architecture
# and to assemble them into the complete model
# The implementation of the transformer architecture is adopted one to
# one from the TensorFlow tutorial "Neural machine translation with a
# Transformer and Keras" (https://www.tensorflow.org/text/tutorials
# /transformer)

# Import required libraries
import numpy as np
import tensorflow as tf


# First, define components required either exclusively by the encoder or
# by both the encoder and the decoder
# Define the `PositionalEmbedding` layer applied to the inputs of both
# the encoder and the decoder
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        # Increment the `max_seq_len` by two in order to take the <START>
        # and <END> token into account
        # As opposed to the original code in the TensorFlow tutorial, the
        # additional argument `max_seq_len` is introduced for the positional embedding
        # This is done with the aim to flexibly adjust the maximum sequence
        # length to the data set currently dealt with
        # In the original code, the maximum sequence length was arbitrarily
        # set to 2048; as this length is exceeded by some sequences, leaving
        # it fixed causes problems
        max_seq_len += 2
        self.pos_encoding = positional_encoding(length=max_seq_len, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


def linear_attention(Q, K, V):
    def feature_map(x):
        return tf.keras.activations.elu(x)  # Simple feature map using ReLU

    Q = feature_map(Q)  # Apply feature map to queries
    K = feature_map(K)  # Apply feature map to keys

    # Compute attention scores
    KV = tf.einsum('...nd,...ne->...de', K, V)
    Z = 1.0 / tf.einsum('...nd,...d->...n', Q, tf.reduce_sum(K, axis=-2))
    output = tf.einsum('...nd,...de,...n->...ne', Q, KV, Z)

    return output


# Define the base attention class, from which more specific attention layers will be branched off
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,dropout =0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert (
                self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)
        self.linear_out = tf.keras.layers.Dense(d_model)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        pass

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]

        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, **kwargs):
        Q = self.split_heads(self.Wq(x))  # Shape: (batch_size, num_heads, seq_len, head_dim)
        K = self.split_heads(self.Wk(x))
        V = self.split_heads(self.Wv(x))

        attn_output = linear_attention(Q, K, V)  # Shape: (batch_size, num_heads, seq_len, head_dim)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # Shape: (batch_size, seq_len, num_heads, head_dim)
        attn_output = tf.reshape(attn_output, tf.shape(x))  # Shape: (batch_size, seq_len, d_model)
        attn_output = self.linear_out(attn_output)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



# Define the feed forward network occurring in both the encoder and the
# decoder
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='elu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


# Assemble the individual encoder components into one encoder layer
# To be more precise, one encoder layer contains a `GlobalSelfAttention`
# layer as well as a `FeedForward` layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = BaseAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


# Assemble the `PositionalEmbedding` layer as well as a defined number
# of `EncoderLayer` layers into the complete encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, max_seq_len, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


# Now, define the residual components required by the decoder
# Define the cross attention layer
class CrossAttention(BaseAttention):
    def call(self, x, context = None):
        if context is None:
            raise ValueError("`context` must be provided for cross attention.")
        Q = self.split_heads(self.Wq(x))
        K = self.split_heads(self.Wk(context))
        V = self.split_heads(self.Wv(context))

        attn_output = linear_attention(Q, K, V)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, tf.shape(x))
        attn_output = self.linear_out(attn_output)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

# Define the causal self-attention layer
class CausalSelfAttention(BaseAttention):
    def call(self, x, context = None):
        Q = self.split_heads(self.Wq(x))
        K = self.split_heads(self.Wk(x))
        V = self.split_heads(self.Wv(x))

        attn_output = linear_attention(Q, K, V)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, tf.shape(x))
        attn_output = self.linear_out(attn_output)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# Assemble the individual decoder components into one decoder layer
# To be more precise, one decoder layer contains one
# `CausalSelfAttention` layer, one `CrossAttention` layer as well as one
# `FeedForward` layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)
        return x


# Assemble the `PositionalEmbedding` layer as well as a defined number
# of `DecoderLayer` layers into one complete decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model,
                                                 max_seq_len=max_seq_len)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, context)
        return x


# Assemble the encoder as well as the decoder into one complete transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=input_vocab_size,
                               max_seq_len=max_seq_len,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=target_vocab_size,
                               max_seq_len=max_seq_len,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            context, x, extra_input = inputs
        else:
            raise ValueError(f"Expected inputs to be a tuple or list with 3 elements, but got {inputs}")

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits
