# The purpose of this Python script is to define the individual
# components constituting the state-of-the-art transformer architecture
# and to assemble them into the complete model
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


# First, define components required either exclusively by the encoder or by both the encoder and the decoder

# `PositionalEmbedding` layer is applied to the inputs of both the encoder and the decoder
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        max_seq_len += 2
        self.pos_encoding = positional_encoding(length=max_seq_len, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x



# Base attention class, from which more specific attention layers will be branched off
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        # Q, K, V projections
        self.query_proj = tf.keras.layers.Dense(d_model)
        self.key_proj = tf.keras.layers.Dense(d_model)
        self.value_proj = tf.keras.layers.Dense(d_model)

        # Linear attention layer
        self.attention = LinearAttention(d_model, num_heads)

        # Normalization and dropout
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking = True

    def split_heads(self, x, batch_size):
        """Split last dim into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine multi-heads back to (batch_size, seq_len, d_model)."""
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)
        seq_len = tf.shape(x)[1]
        return tf.reshape(x, (batch_size, seq_len, self.d_model))

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]

        # Compute Q, K, V
        query = self.query_proj(x)  # (batch_size, seq_len, d_model)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Split heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Apply linear attention
        attn_output = self.attention(query, key, value, mask)  # (batch_size, num_heads, seq_len, depth)

        # Combine heads
        attn_output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)
        attn_output = self.dropout(attn_output)

        # Residual connection and layer norm
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class LinearAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(LinearAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Q, K, V projection layers
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

        # Output projection
        self.out_dense = tf.keras.layers.Dense(d_model)

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """(batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x, batch_size):
        """(batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, d_model)"""
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        # Linear projections for Q, K, V
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads for multi-head attention
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Feature map transformation (ELU + 1)
        query = tf.nn.elu(query) + 1
        key = tf.nn.elu(key) + 1

        # Compute key-value product (Linear Attention core)
        kv_product = tf.einsum('bhld,bhmd->bhlm', key, value)  # (batch, heads, depth, depth)
        query_key_sum = tf.einsum('bhld,bhld->bhl', query, key) + 1e-6  # (batch, heads, seq_len)

        # Attention output
        output = tf.einsum('bhlm,bhld->bhmd', kv_product, query)  # (batch, heads, seq_len, depth)
        output = output / query_key_sum[..., tf.newaxis]  # Normalization

        # Combine heads
        output = self.combine_heads(output, batch_size)

        # Output projection
        output = self.out_dense(output)

        # Apply dropout
        output = self.dropout(output)

        return output


# Feed forward network occurring in both the encoder and the decoder
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


# Assemble the individual encoder components into one encoder layer
# One encoder layer contains a `GlobalSelfAttention` layer as well as a `FeedForward` layer
class EncoderLayer(BaseAttention):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__(d_model=d_model, num_heads=num_heads)
        self.self_attention = LinearAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        self.supports_masking = True

    def call(self, x, mask=None):
        # Linear attention output
        attn_output = self.self_attention(query=x, key=x, value=x, mask=mask)

        # Add and normalize
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        # Feedforward network
        x = self.ffn(x)
        return x


# Assemble the `PositionalEmbedding` layer as well as a defined number `EncoderLayer` layers into the complete encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking = True

    def call(self, x, mask=None):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)

        return x  # Shape `(batch_size, seq_len, d_model)`.


# Residual components required by the decoder

# Cross attention layer
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output = self.attention(
            query=x,
            key=context,
            value=context,
            #return_attention_scores = True
        )

        # Cache the attention scores for plotting later.
        #self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# Causal self-attention layer
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# Assemble the individual decoder components into one decoder layer
# One decoder layer contains one `CausalSelfAttention` layer, one `CrossAttention` layer and one `FeedForward` layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        self.supports_masking = True

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        #self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


# Assemble the `PositionalEmbedding` layer and a defined number of `DecoderLayer` layers into one complete decoder
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
        self.supports_masking = True

    def call(self, x, context):
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        #self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


# Assemble the encoder and the decoder into one complete transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               max_seq_len=max_seq_len,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               max_seq_len=max_seq_len,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        print("Transformer `call` started")
        context, x = inputs
        print(f"Context shape: {context.shape}")
        print(f"Target shape: {x.shape}")
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        print(f"Context after encoder: {context.shape}")

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        print(f"Output after decoder: {x.shape}")
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
