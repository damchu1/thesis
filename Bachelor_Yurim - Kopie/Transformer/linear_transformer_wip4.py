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
    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_causal_mask = False):
        super().__init__()
        self.attention = LinearAttention(d_model=d_model, num_heads=num_heads, use_causal_mask=use_causal_mask)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.supports_masking = True

    def call(self, query, key, value, use_causal_mask=False):
        attn_output = self.attention(query, key, value,use_causal_mask=use_causal_mask)
        x = self.add([query, attn_output])
        return self.layernorm(x)


class LinearAttention(tf.keras.layers.Layer):
    """

    Implements a linear-attention mechanism with an optional causal (prefix-sum)
    mode. For the causal mode, we do a sequential prefix-sum scan so that each
    position can only attend to its past. This is O(L * d) per head—slower than
    parallel prefix methods, but simpler to read and correct for causal logic.

    NOTE: This example focuses on self-attention. For cross-attention or more
    advanced usage, you'll need to adapt it further.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1, use_causal_mask=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.depth = d_model // num_heads
        self.query_proj = tf.keras.layers.Dense(d_model)
        self.key_proj = tf.keras.layers.Dense(d_model)
        self.value_proj = tf.keras.layers.Dense(d_model)
        self.out_proj = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, query, key, value, training=None,use_causal_mask=False):

        """
        query, key, value: (batch_size, seq_len, d_model)
        """

        batch_size = tf.shape(query)[0]

        seq_len = tf.shape(query)[1]  # assume Q, K, V all have same seq_len

        # Linear layers to get Q, K, V

        Q = self.query_proj(query)  # (B, L, d_model)

        K = self.key_proj(key)  # (B, L, d_model)

        V = self.value_proj(value)  # (B, L, d_model)

        # Split into heads

        # After split_heads: (B, H, L, depth)

        Q = self.split_heads(Q, batch_size)

        K = self.split_heads(K, batch_size)

        V = self.split_heads(V, batch_size)

        # Optional dropout on Q/K/V (uncomment if desired)
        # Q = self.dropout(Q, training=training)
        # K = self.dropout(K, training=training)
        # V = self.dropout(V, training=training)
        # Feature map: phi(x) = ELU(x) + 1

        Q = tf.nn.elu(Q) + 1.0

        K = tf.nn.elu(K) + 1.0

        if not self.use_causal_mask:

            # ------------------------
            # Non-causal linear attention
            # ------------------------
            #

            # Formula (one possible variant):
            #   output = (Q * (K^T V)) / (K^T 1_{Q})
            # but we code it carefully with einsum.
            # (B, H, L, depth) x (B, H, L, depth) -> (B, H, L, depth)
            # For each of the L positions, we compute sum over "l" dimension:
            #   kv_product = sum_{l} [K_l * V_l]

            kv_product = tf.einsum('bhld,bhmd->bhlm', K, V)

            # shape is (B, H, L, M), but typically M == depth if the time dims match.
            # Actually let's keep them in consistent dimension naming:
            #   K: (B, H, L, D)
            #   V: (B, H, L, D)
            # => kv_product: (B, H, L, D)
            # Z = K^T (sum Q)
            #   => reduce_sum(Q, axis=2) => shape (B, H, D)
            #   => multiply with K => shape (B, H, L) after einsum

            sumQ = tf.reduce_sum(Q, axis=2)  # (B, H, depth)

            Z = tf.einsum('bhld,bhd->bhl', K, sumQ)  # (B, H, L)

            Z = tf.expand_dims(Z, axis=-1)  # => (B, H, L, 1)

            # Then multiply kv_product with Q:
            #   => shape (B, H, L, D)
            # Finally divide by Z

            output = tf.einsum('bhlm,bhld->bhmd', kv_product, Q) / (Z + 1e-9)



        else:

            # ------------------------
            # Causal linear attention
            # ------------------------
            #
            # We implement the prefix-sum approach in a single pass:
            #   For i-th query:
            #     denom = Q_i · sum_{j <= i}(K_j)
            #     out_i = Q_i * sum_{j <= i}(K_j * V_j) / denom
            #

            # We'll do this per head in a loop-like fashion using tf.scan,
            # which is simpler to read (though not the fastest possible).
            # Merge batch & head dims => shape: (B*H, L, depth)

            BH = batch_size * self.num_heads

            Q_2d = tf.reshape(Q, [BH, seq_len, self.depth])

            K_2d = tf.reshape(K, [BH, seq_len, self.depth])

            V_2d = tf.reshape(V, [BH, seq_len, self.depth])

            # Transpose to (L, BH, depth) so we can scan over L

            Q_t = tf.transpose(Q_2d, [1, 0, 2])  # (L, BH, D)

            K_t = tf.transpose(K_2d, [1, 0, 2])

            V_t = tf.transpose(V_2d, [1, 0, 2])

            def scan_fn(carry, inputs):

                """

                carry: (K_cum, KV_cum), each shape (BH, D)
                inputs: (Q_i, K_i, V_i) for a single time-step i
                        each shape (BH, D)

                """

                K_cum, KV_cum = carry

                Q_i, K_i, V_i = inputs

                # denom = Q_i dot K_cum  (shape: (BH,))

                denom = tf.reduce_sum(Q_i * K_cum, axis=-1, keepdims=True)  # (BH, 1)

                # out_i = Q_i * KV_cum / denom (elementwise over BH x D)

                out_i = (Q_i * KV_cum) / (denom + 1e-9)

                # Update prefix sums

                K_cum_new = K_cum + K_i

                KV_cum_new = KV_cum + (K_i * V_i)

                return (K_cum_new, KV_cum_new), out_i

            # Initial carry: all zeros

            K_cum_init = tf.zeros([BH, self.depth], dtype=Q.dtype)

            KV_cum_init = tf.zeros([BH, self.depth], dtype=Q.dtype)

            # Pack (Q_t, K_t, V_t) => each step we get (Q_i, K_i, V_i)

            scan_elems = (Q_t, K_t, V_t)

            # tf.scan returns (final_state, stacked_outputs)

            # stacked_outputs will have shape (L, BH, depth)

            _, outputs_t = tf.scan(

                fn=scan_fn,

                elems=scan_elems,

                initializer=(K_cum_init, KV_cum_init),

                parallel_iterations=1  # ensure true sequential

            )

            # outputs_t => (L, BH, D). We transpose back => (BH, L, D)

            outputs = tf.transpose(outputs_t, [1, 0, 2])

            # Reshape back to (B, H, L, depth)

            output = tf.reshape(outputs, [batch_size, self.num_heads, seq_len, self.depth])

        # (Optional) dropout on the attention output

        # output = self.dropout(output, training=training)

        # Combine heads back to (B, L, d_model)

        output = self.combine_heads(output, batch_size)

        # Final linear projection

        output = self.out_proj(output)

        return output

    def split_heads(self, x, batch_size):
        """ Reshape (B, L, d_model) -> (B, num_heads, L, depth). """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x, batch_size):
        """ Inverse of split_heads: (B, num_heads, L, depth) -> (B, L, d_model). """
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (B, L, H, depth)
        return tf.reshape(x, (batch_size, -1, self.d_model))


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
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = LinearAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        self.supports_masking = True

    def call(self, x):
        x = self.self_attention(query=x, key=x, value=x)
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

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


# Residual components required by the decoder

# Cross attention layer
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output = self.attention(
            query=x, key=context, value=context
            #return_attention_scores = True
        )

        # Cache the attention scores for plotting later.
        #self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# Causal self-attention layer
class CausalSelfAttention(BaseAttention):
    def call(self, x, use_causal_mask = False):
        attn_output = self.attention(
            query=x, key=x, value=x, use_causal_mask= use_causal_mask
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
        x = self.causal_self_attention(x)
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
