# This Python script serves the purpose of performing an evaluation of
# the training data set with the optimal weights obtained at the end of
# the training procedure without dropout
# This is necessary in order to be able to compare the accuracy on the
# training data set to the one on the validation data set, as dropout is
# turned off when evaluating the validation data set
# Hence, when the accuracy on the training data set that was obtained
# during training is compared to the validation accuracy, it is lower,
# which is counterintuitive and raises questions

# Instead of employing 'model.evaluate' for the evaluation, the
# comparison of the prediction with the expected output and the
# computation of the accuracy are performed manually
# This is due to the fact that 'model.evaluate' cannot be applied to
# autoregressive models, as is the case with the Transformer

import pickle

import numpy as np
from tensorflow import keras
from mpi4py import MPI

from Transformer.auxiliary_file import train_validate_test_split
from Transformer.tokenisation import N_AA_TOKENS, N_CODON_TOKENS,\
    format_dataset, translate_aa_into_nt,\
        tokenise_nt_seqs_for_eval
from Transformer.transformer import Transformer

comm = MPI.COMM_WORLD
# The variable 'size' stores the total amount of processes, i.e. the
# total amount of available cores, whereas the variable 'rank' harbours
# integers specifying those particular processes/cores
size = comm.Get_size()
rank = comm.Get_rank()

# The constant 'MAX_SEQ_LEN'must be adjusted to the data set currently dealt with!
# In the case of the E. coli data set comprising six genomes, the
# maximum sequence length was set to 500
# Hence, including a safety buffer, the constant is set to 1000
MAX_SEQ_LENGTH = 1000

# Only the training set is required, but neither the validation set nor the test set
training_set, _, _ = train_validate_test_split(
    path_to_csv="/work/scratch/ja63nuwu/E_coli_6_genomes_aa_nt_seqs.csv",
    train_size=0.8,
    valid_size=0.1
)

# The instantiation of the transformer architecture is conducted for all ranks
# Load the weights obtained from training
weights_file_path = (
    f"/work/scratch/ja63nuwu/E-coli-6-genomes-trained-weights_6th_approach.pkl"
)

with open(weights_file_path, "rb") as f:
    model_weights, _ = pickle.load(f)

# Define the Transformer hyperparameters
num_layers = 3
d_model = 128
dff = 512
num_heads = 4
dropout_rate = 0.1

keras.backend.clear_session()

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=N_AA_TOKENS,
    target_vocab_size=N_CODON_TOKENS,
    max_seq_len=MAX_SEQ_LENGTH,
    dropout_rate=dropout_rate
)

# Now, set the weights
# Until now, the model was merely instantiated, but not built
# In order to be able to set the weights, the model must be built, which
# can be achieved by running the model once with some dummy input
# The dummy input is kept as small as possible in order not to waste time
# Slicing is done instead of direct indexing for retrieving a dummy
# sequence pair, as the latter results in a dimensionality reduction,
# which in turn causes an error within the function
# 'format_dataset_for_predictive_alignment'
dummy_seq_pair = training_set[:1]
(
    dummy_aa_encoding, dummy_nt_encoding, actual_seq_lens
), _ = format_dataset(dummy_seq_pair)

# Finally, run the Transformer on the dummy input
transformer((dummy_aa_encoding, dummy_nt_encoding, actual_seq_lens))

# Setting the weights is now hopefully possible
transformer.set_weights(model_weights)

if rank == 0:
    # Convert the Pandas DataFrame to a NumPy array
    training_set_array = training_set.to_numpy()
    aa_seqs, ref_nt_seqs = zip(*training_set_array)

    # The built-in zip function returns an iterator of tuples; those tuples are converted to lists
    aa_seqs = list(aa_seqs)
    ref_nt_seqs = list(ref_nt_seqs)

    # It must be kept in mind that when employing the scatter function,
    # the amount of elements to be scattered across the ranks must
    # exactly equal the total amount of ranks, i.e. the amount of processes
    # In order to ensure this, the entire training data set of amino
    # acid sequences is evenly split into chunks, whereby the total
    # amount of chunks equals the total amount of available ranks/processes
    aa_seq_chunks = [[] for _ in range(size)]
    # In order to distribute the sequences across all processes, a
    # trick involving the modulo operator is employed
    for i, aa_seq in enumerate(aa_seqs):
        aa_seq_chunks[i % size].append(aa_seq)
    
    # Nucleotide sequences, which serve as reference sequences in the
    # quantitative evaluation, are split into chunks as well
    ref_nt_seq_chunks = [[] for _ in range(size)]
    for i, nt_seq in enumerate(ref_nt_seqs):
        ref_nt_seq_chunks[i % size].append(nt_seq)
else:
    # For some peculiar reason, beyond the variables to which the
    # sequences are distributed, also the variable from which the
    # distribution takes place needs to be declared in all ranks
    aa_seq_chunks = None
    aa_seqs = None

    ref_nt_seq_chunks = None
    ref_nt_seqs = None

aa_seqs = comm.scatter(aa_seq_chunks, root=0)
ref_nt_seqs = comm.scatter(ref_nt_seq_chunks, root=0)

# Run inference for each of the individual processors
# Note that the training data set is quite large as it receives 80% of the total data set
# Hence, without taking any further steps, each processor would need to
# handle a significant amount of sequences, which results in an OOM error
# Instead, the sequences need to be manually batched into batches of
# size 5 (it was empirically determined that a batch size of 5 prevents
# an OOM error for this particular job)
# However, this is taken care of in the function 'translate_aa_into_nt'
raw_translated_nt_seqs = translate_aa_into_nt(
    transformer, aa_seqs, MAX_SEQ_LENGTH, return_string=False, batch_size=5
)

# Now, compute for each process the amount of predictions matching their
# respective label as well as the total amount of labels
# To this end, the reference nucleotide sequences first need to be tokenised
tokenised_labels = tokenise_nt_seqs_for_eval(
    ref_nt_seqs, max_seq_len=MAX_SEQ_LENGTH
)
matching_preds_and_labels = raw_translated_nt_seqs == tokenised_labels

# For both quantities, i.e. the amount of predictions matching their
# label and the total amount of labels, the padding token zero is neglected
# For this purpose, a mask is generated and applied to the array
# harbouring the matches between predictions and labels
padding_mask = tokenised_labels != 0
matching_preds_and_labels = matching_preds_and_labels & padding_mask

# Finally, compute the amount of  predictions matching their labels as
# well as the total amount of labels for each process
n_matches = np.sum(matching_preds_and_labels)
n_labels = np.sum(padding_mask)

# Now, gather the amount of predictions matching their labels as well as
# the total amount of labels from all processes in rank 0
# 'comm.gather' returns a list comprising the gathered objects
n_matches = comm.gather(n_matches, root=0)
n_labels = comm.gather(n_labels, root=0)

# Compute the sum of both lists
if rank == 0:
    sum_matches = np.sum(n_matches)
    sum_labels = np.sum(n_labels)
    accuracy = sum_matches / sum_labels
    print(
        "The accuracy of the Transformer on the training data set "
        f"without dropout is {accuracy}."
    )