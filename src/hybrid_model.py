"""Keras Transformer model for token edits and punctuation restoration."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class TokenCandidateEmbedding(layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int,
        punct_classes: int = 8,
        candidate_top_k: int = 5,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        self.punct_classes = punct_classes
        self.candidate_top_k = max(1, int(candidate_top_k))
        self.dropout_rate = dropout_rate
        self.token_embedding = layers.Embedding(vocab_size, d_model, name="token_embedding")
        self.candidate_embedding = layers.Embedding(vocab_size, d_model, name="candidate_embedding")
        self.candidate_rank_embedding = layers.Embedding(self.candidate_top_k, d_model, name="candidate_rank_embedding")
        self.candidate_projection = layers.Dense(d_model, name="candidate_projection")
        self.source_punct_embedding = layers.Embedding(punct_classes, d_model, name="source_punct_embedding")
        self.position_embedding = layers.Embedding(max_length, d_model, name="position_embedding")
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        if len(inputs) == 2:
            token_ids, candidate_ids = inputs
            source_punct_ids = tf.zeros_like(token_ids)
        else:
            token_ids, candidate_ids, source_punct_ids = inputs
        length = tf.shape(token_ids)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        x = self.token_embedding(token_ids)
        if len(candidate_ids.shape) == 2:
            candidate_ids = tf.expand_dims(candidate_ids, axis=-1)
            candidate_ids = tf.tile(candidate_ids, [1, 1, self.candidate_top_k])
        candidate_emb = self.candidate_embedding(candidate_ids)
        rank_ids = tf.range(start=0, limit=self.candidate_top_k, delta=1)
        rank_emb = self.candidate_rank_embedding(rank_ids)
        candidate_emb = candidate_emb + rank_emb
        candidate_shape = tf.shape(candidate_emb)
        candidate_flat = tf.reshape(
            candidate_emb,
            [candidate_shape[0], candidate_shape[1], self.candidate_top_k * self.d_model],
        )
        x = x + self.candidate_projection(candidate_flat)
        x = x + self.source_punct_embedding(source_punct_ids)
        x = x + self.position_embedding(positions)
        x = self.layer_norm(x)
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "d_model": self.d_model,
                "punct_classes": self.punct_classes,
                "candidate_top_k": self.candidate_top_k,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu"),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model),
            ],
            name="ffn",
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def build_hybrid_model(
    vocab_size: int,
    max_length: int,
    action_classes: int,
    punct_classes: int,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-4,
    candidate_top_k: int = 5,
) -> keras.Model:
    token_ids = layers.Input(shape=(max_length,), dtype="int32", name="token_ids")
    candidate_ids = layers.Input(shape=(max_length, candidate_top_k), dtype="int32", name="candidate_ids")
    source_punct_ids = layers.Input(shape=(max_length,), dtype="int32", name="source_punct_ids")

    x = TokenCandidateEmbedding(vocab_size, max_length, d_model, punct_classes, candidate_top_k, dropout_rate)(
        [token_ids, candidate_ids, source_punct_ids]
    )
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"encoder_block_{i + 1}",
        )(x)

    shared = layers.Dropout(dropout_rate, name="shared_dropout")(x)
    action = layers.Dense(action_classes, activation="softmax", name="action")(shared)
    punct = layers.Dense(punct_classes, activation="softmax", name="punct")(shared)

    model = keras.Model(
        inputs={
            "token_ids": token_ids,
            "candidate_ids": candidate_ids,
            "source_punct_ids": source_punct_ids,
        },
        outputs={"action": action, "punct": punct},
        name="HybridEditPunctuationTransformer",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5, clipnorm=1.0),
        loss={
            "action": keras.losses.SparseCategoricalCrossentropy(),
            "punct": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={"action": 1.0, "punct": 0.35},
        metrics={
            "action": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "punct": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
        },
    )
    return model
