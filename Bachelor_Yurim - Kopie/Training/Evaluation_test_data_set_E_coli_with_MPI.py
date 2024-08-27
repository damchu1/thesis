import pandas as pd
import pickle
import collections

import numpy as np
from tensorflow import keras
from mpi4py import MPI

from auxiliary_file import train_validate_test_split, _count_codons, \
    compute_norm_levenshtein
from tokenisation import N_AA_TOKENS, N_CODON_TOKENS, \
    format_dataset, translate_aa_into_nt
from transformer_wip import Transformer

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# The constant `MAX_SEQ_LEN` must be adjusted to the data set currently dealt with!
# In the case of the E. coli data set comprising six genomes, the
# maximum sequence length was set to 500
# Hence, including a safety buffer, the constant is set to 1000
MAX_SEQ_LENGTH = 1000

_, _, test_set = train_validate_test_split(
    path_to_csv="/work/scratch/ja63nuwu/E_coli_6_genomes_aa_nt_seqs.csv",
    train_size=0.8,
    valid_size=0.1
)

# The instantiation of the transformer architecture is conducted for all ranks
# Load the weights obtained from training
weights_file_path = (
    "/work/scratch/ja63nuwu/E-coli-6-genomes-trained-weights_6th_approach_window_width_51.pkl"
)

with open(weights_file_path, "rb") as f:
    model_weights, _ = pickle.load(f)

# Define the transformer hyperparameters
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
dummy_seq_pair = test_set[:1]
(
    dummy_aa_encoding, dummy_nt_encoding, actual_seq_lens
), _ = format_dataset(dummy_seq_pair)

# Finally, run the transformer on the dummy input
transformer((dummy_aa_encoding, dummy_nt_encoding, actual_seq_lens))

# Setting the weights is now hopefully possible
transformer.set_weights(model_weights)

if rank == 0:
    test_set_array = test_set.to_numpy()
    aa_seqs, ref_nt_seqs = zip(*test_set_array)

    aa_seqs = list(aa_seqs)
    ref_nt_seqs = list(ref_nt_seqs)

    # It must be kept in mind that when employing the scatter
    # function, the amount of elements to be scattered across the
    # ranks must exactly equal the total amount of ranks, i.e. the amount of processes
    # In order to ensure this, the entire test data set of amino
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
    aa_seq_chunks = None
    aa_seqs = None

    ref_nt_seq_chunks = None
    ref_nt_seqs = None

aa_seqs = comm.scatter(aa_seq_chunks, root=0)
ref_nt_seqs = comm.scatter(ref_nt_seq_chunks, root=0)

# Run inference for each of the individual processors
raw_translated_nt_seqs = translate_aa_into_nt(
    transformer, aa_seqs, MAX_SEQ_LENGTH
)

# Compute the Levenshtein distance for the individual prediction-reference pairs
# Additionally, create a Counter object in order to track the amount of times individual codons occur
# This is done in order to compare the codon usage bias learned by the
# transformer to the frequencies reported in literature
levenshtein_list = []
codon_counter = collections.Counter()
codon_counter_list = None
for raw_translated_nt_seq, ref_nt_seq in zip(
        raw_translated_nt_seqs, ref_nt_seqs
):
    translated_nt_seq = (
        raw_translated_nt_seq.replace("<PAD>", "")
        .replace("<START>", "")
        .replace("<END>", "")
        .strip()
    )

    # Count the amount of times the individual codons appear and update the global Counter object
    codon_count_dict = _count_codons(translated_nt_seq)
    codon_counter.update(codon_count_dict)

    current_levenshtein_distance = compute_norm_levenshtein(
        translated_nt_seq, ref_nt_seq
    )

    levenshtein_list.append(current_levenshtein_distance)

# Gather all values for the Levenshtein distance in rank 0
levenshtein_list = comm.gather(levenshtein_list, root=0)

# Apart from that, also gather all Counter objects in rank 0
codon_counter_list = comm.gather(codon_counter, root=0)

if rank == 0:
    # The gathering process yields a list of lists, i.e. a nested list
    # Hence, the individual sublists need to be merged into one list
    # The list comprehension below is the pythonic way of accomplishing this task
    # It can be regarded as the shorthand notation for a nested for-loop
    # (for more details, search for "python flatten nested list" in Google)
    levenshtein_list = [
        value for sublist in levenshtein_list
        for value in sublist
    ]

    # Now, compute the quantities required for the box-and-whisker plot
    # The required quantities comprise the minimum, the maximum, the
    # median, the first quartile and the third quartile
    min_lv_dist = min(levenshtein_list)
    max_lv_dist = max(levenshtein_list)
    median_lv_dist = np.median(levenshtein_list)
    first_quartile_lv_dist = np.percentile(levenshtein_list, 25)
    third_quartile_lv_dist = np.percentile(levenshtein_list, 75)

    # Additionally, the mean value and the standard deviation are computed
    mean_lv_dist = np.mean(levenshtein_list)
    std_dev_lv_dist = np.std(levenshtein_list)

    # Output the results to the screen
    metrics_text = (
        f"The minimum Levenshtein distance is {min_lv_dist:.4f}.\n"
        f"The maximum Levenshtein distance is {max_lv_dist:.4f}.\n"
        f"The median of the Levenshtein distance is {median_lv_dist:.4f}.\n"
        f"The first quartile of the Levenshtein distance is "
        f"{first_quartile_lv_dist:.4f}.\n"
        f"The third quartile of the Levenshtein distance is "
        f"{third_quartile_lv_dist:.4f}.\n\n"
        f"The mean Levenshtein distance is {mean_lv_dist:.4f}.\n"
        f"The standard deviation is {std_dev_lv_dist:.4f}."
    )
    print(metrics_text)

    # Furthermore, save the results in a text file
    with open(
            "Levenshtein_distance_metrics_E_coli_6_genomes_6th_approach_window_width_51.txt", "w"
    ) as f:
        f.write(metrics_text)

    # Finally, collect the codon counts from the individual processors
    # and compute the codon usage frequencies
    global_codon_counter = collections.Counter()
    for processor_counter in codon_counter_list:
        global_codon_counter.update(processor_counter)

    # Compute the sum of all values in the Counter
    codon_count_sum = sum(codon_counter.values())

    for key, value in codon_counter.items():
        # Codon usage frequencies are commonly given in per mille
        codon_counter[key] = (value / codon_count_sum) * 1000

    # Print the result to the screen
    print(codon_counter)

    df = pd.DataFrame(codon_counter.items(), columns=['item', 'count'])

    df.to_csv(f'Codon_usage_data/codon_usage_E_coli.csv', index=False)
    print('Codon usage data successfully saved!')
