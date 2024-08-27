import numpy as np
import tensorflow as tf
from keras.src.layers import MultiHeadAttention

from Transformer.tokenisation import format_dataset, N_AA_TOKENS, N_CODON_TOKENS
from Transformer.auxiliary_file import train_validate_test_split
from Transformer.transformer_wip import PositionalEmbedding

# The largest sequence length of all the sequences in the entire data
# set must be defined for the positional encoding
max_seq_len = 1000

training_set, validation_set, test_set = train_validate_test_split(
    path_to_csv=r"C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\Transformer\e_coli_aa_nt_seqs.csv",
    train_size=0.8,
    valid_size=0.1
)

first_seq_pairs = training_set[:3]
(aa_encoding, nt_input_encoding), nt_output_encoding = format_dataset(first_seq_pairs)

# Perform the embedding of the integer tokens in conjunction with positional encoding
embedding = PositionalEmbedding(vocab_size=N_AA_TOKENS, d_model=12, max_seq_len=max_seq_len)
embedded_tokens = embedding(aa_encoding)


mha_layer = MultiHeadAttention(
    num_heads=3,
    key_dim=512
)

mha_layer(query=embedded_tokens, value=embedded_tokens)