# The purpose of this Python script is to define the functions required
# for tokenising amino acid as well as nucleotide sequences

from math import ceil

import tensorflow as tf
import numpy as np

# Tokenisation must be performed for the "language" of amino acids as well as for the "language" of nucleotides
# For both languages, tokenisation is performed via mapping of the individual tokens to integer indices

# Start with the preparation of amino acid tokenisation
# The enumeration of amino acids below does not contain selenocysteine
# (U) and the letter X, denoting any amino acid / an unknown amino acid
# This is due to the fact that the coding sequences were downloaded from
# NCBI, not the protein sequences, and nucleotide sequences containing
# ambiguous characters (such as N for any nucleotide) were excluded
# Hence, the amino acid character X does not occur
# Moreover, the default table from NCBI was used for translation, i. e.
# the amino acid selenocysteine does not occur in the protein sequences
# However, the asterisk is included, as it denotes a stop codon
ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY*'
# Compared to the tokenisation implemented in ProteinBERT, the '<OTHER>'
# token is omitted as an entirely unambiguous alphabet is employed for
# both the nucleotide sequences and the amino acid sequences
# The padding token is usually mapped to the integer zero, which is why
# this is done here as well
# The consequences of mapping the padding token to another integer or
# whether this has any consequences at all is not known
ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>']

# To each sequence, one <START> and <END> token is added, respectively
ADDED_TOKENS_PER_SEQ = 2

n_aas = len(ALL_AAS)
n_add_aa_tokens = len(ADDITIONAL_TOKENS)
aa_token_to_index = {
    aa: i + n_add_aa_tokens for i, aa in enumerate(ALL_AAS)
}
additional_aa_token_to_index = {
    token: i for i, token in enumerate(ADDITIONAL_TOKENS)
}
global_aa_token_to_index = {
    **additional_aa_token_to_index, **aa_token_to_index
}
index_to_aa_token = {
    index: token for token, index in global_aa_token_to_index.items()
}
N_AA_TOKENS = len(global_aa_token_to_index)


def _parse_seq(seq):
    """
    This function is responsible for converting an input sequence to a
    string, if applicable, and to return the input string otherwise.

    Parameters
    ----------
    seq: str or binary
        The input amino acid sequence to be tokenised.

    Returns
    -------
    seq: str
        The input amino acid sequence to be tokenised.
    """
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, bytes):
        return seq.decode('utf8')
    else:
        raise TypeError('Unexpected sequence type: %s' % type(seq))


def _tokenise_aa_seq(seq):
    """
    This function performs tokenisation for amino acid sequences. This
    is achieved by splitting an amino acid sequence into individual
    tokens, i.e. individual amino acids, and replacing all individual
    tokens with their unique integer index defined in the dictionary
    `token_to_index`.

    Apart from that, one <START> and <END> token is added to each
    sequence, respectively. The token dictionary also comprises a
    special token, <OTHER>, reserved for unknown amino acids.

    This function employs the private function `_parse_seq`.

    Parameters
    ----------
    seq:  str or binary
        The amino acid sequence to be tokenised.

    Returns
    -------
    tokenised_seq: list, dtype=int
        The tokenised analogue of the input amino acid sequence.
    """
    # It is important to prepend the <START> token and append <END> token to the sequence
    tokenised_seq = (
            [additional_aa_token_to_index["<START>"]]
            +
            [
                aa_token_to_index.get(aa) for aa in _parse_seq(seq)
            ]
            +
            [additional_aa_token_to_index["<END>"]]
    )

    return tokenised_seq


def _tokenise_aa_seqs(seqs):
    """
    This function performs tokenisation for a whole data set of amino
    acid sequences.

    For this purpose, it employs the private function
    `_tokenise_aa_seq`.

    Parameters
    ----------
    seqs: iterable, dtype=str or dtype=binary
        An iterable containing the amino acid sequences to be tokenised.

    Returns
    -------
    tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
        A NumPy array harbouring the tokenised analogues of the input
        amino acid sequences. The first dimension, m, corresponds to the
        amount of input sequences, whereas the second dimension, n,
        corresponds to the largest sequence length in the batch.
    """
    # First, perform the mere tokenisation
    tokenised_seqs_list = []

    for seq in seqs:
        tokenised_seqs_list.append(_tokenise_aa_seq(seq))

    # Now, compute the largest length of the tokenised sequences
    # comprised in the batch
    # Computing the largest length of the tokenised (integer) sequences
    # instead of the plain string sequences is necessary as in the
    # course of tokenization, the length of each sequence is incremented
    # by two due to the addition of the <START> and <END> token
    max_length = max(map(len, tokenised_seqs_list))

    # Perform padding for each of the tokenised sequences
    tokenised_seqs_array = np.array(
        [tokenised_seq
         +
         (max_length - len(tokenised_seq)) * [additional_aa_token_to_index['<PAD>']]
         for tokenised_seq in tokenised_seqs_list if tokenised_seq],
        dtype=np.int32
    )

    return tokenised_seqs_array


# Now, prepare nucleotide/codon tokenisation
# As it is exclusively dealt with DNA sequences, and not RNA sequences,
# uracil is omitted
# The letter N, denoting any nucleotide, is not included as sequences
# containing ambiguous nucleotides were discarded in advance in order to
# permit an unambiguous translation into amino acids
# Note that in the case of nucleotide sequences, tokenisation is not
# performed for the individual characters/nucleotides as with the amino
# acids, but rather for individual codons, i. e. triplets
ALL_CODONS = []
for nt_1 in "ATCG":
    for nt_2 in "ATCG":
        for nt_3 in "ATCG":
            ALL_CODONS.append(nt_1 + nt_2 + nt_3)

# As with the amino acid tokenisation, the '<OTHER>' token is omitted as
# an entirely unambiguous alphabet is used for the nucleotide
# tokenisation and sequences containing ambiguous characters have been
# discarded
# Again, the padding token is deliberately mapped to the integer zero as
# the consequences of deviating from this convention are not known
ADDITIONAL_NT_TOKENS = ['<PAD>', '<START>', '<END>']

n_codons = len(ALL_CODONS)
n_additional_tokens = len(ADDITIONAL_NT_TOKENS)
codon_token_to_index = {
    codon: i + n_additional_tokens for i, codon in enumerate(ALL_CODONS)
}
additional_codon_token_to_index = {
    token: i for i, token in enumerate(ADDITIONAL_NT_TOKENS)
}
global_codon_token_to_index = {
    **additional_codon_token_to_index, **codon_token_to_index
}
index_to_codon_token = {
    index: token for token, index in global_codon_token_to_index.items()
}
N_CODON_TOKENS = len(global_codon_token_to_index)


def _split_into_triplets(nt_seq):
    """
    This function splits an input nucleotide sequence into its
    individual codons / triplets.

    The function assumes that the sequence length is a multiple of three
    and that the first and last codon is the AUG start codon and one of
    the three stop codons, respectively.

    Parameters
    ----------
    nt_seq: str
        The nucleotide sequence to be split into triplets / codons.

    Returns
    -------
    codons: list, dtype=str
        The list containing the codons the input nucleotide sequence is
        made up of.
    """
    codons = []
    n_triplets = len(nt_seq) // 3

    for i in range(n_triplets):
        start_index = 3 * i
        end_index = 3 * (i + 1)
        triplet = nt_seq[start_index:end_index]
        codons.append(triplet)

    return codons


def _split_into_nt_frags(tokenised_seq, frag_len, terminal_cutoff):
    """
    This function...

    Parameters
    ----------
    tokenised_seq: pass
        pass
    frag_len: int
        pass
    terminal_cutoff: int
        pass

    Returns
    -------
    fragment_list: list, dtype=str
        pass
    """
    pure_seq_len = frag_len - 2

    n_fragments = ceil(len(tokenised_seq) / pure_seq_len)

    # Perform the splitting by iterating over the list
    fragment_list = []

    start_index = 0
    end_index = pure_seq_len

    for _ in range(n_fragments):
        current_fragment = tokenised_seq[start_index:end_index]
        # Prepend the <START> token and append the <END> token to the
        # fragment
        current_fragment = (
                [additional_codon_token_to_index["<START>"]]
                +
                current_fragment
                +
                [additional_codon_token_to_index["<END>"]]
        )

        # Increase the indices in order to obtain the next fragment in
        # the next iteration step
        start_index += pure_seq_len
        end_index += pure_seq_len

        # Decide whether to use or to discard the current fragment
        if len(current_fragment) >= terminal_cutoff:
            fragment_list.append(current_fragment)
        # Taking into account the case that one complete coding sequence
        # is shorter than the desired fragment length
        elif n_fragments == 1:
            fragment_list.append(current_fragment)
        else:
            continue

    return fragment_list


def split_nt_seqs_for_eval(nt_seqs, frag_len, terminal_cutoff):
    """
    This function...

    Parameters
    ----------
    nt_seqs: list, dtype=str
        pass
    frag_len: int
        pass
    terminal_cutoff: int
        pass

    Returns
    -------
    frags_list: list, dtype=str
        pass
    """
    # In order to obtain the amount of nucleotides to be included in one
    # fragment, the fragment length provided as input must be reduced by
    # two and the result must be multiplied by three
    # The first operation is necessary as the input fragment length
    # already comprises the <START> and <END> tokens
    # The second operation is necessary as the input fragment length
    # refers to the amount of tokens, whereas three nucleotides yield
    # one token
    pure_seq_len = (frag_len - 2) * 3

    # Due to the aforementioned reasons, the quantity `terminal_cutoff`
    # needs to be reduced by two and the result needs to be multiplied
    # by three as well
    terminal_cutoff = (terminal_cutoff - 2) * 3

    # The statement below defines the list storing fragments from all
    # nucleotide sequences
    frags_list = []

    # Iterate over the individual nucleotide sequences provided as input
    for nt_seq in nt_seqs:
        n_fragments = ceil(len(nt_seq) / pure_seq_len)

        # Iterate over the nucleotide sequence currently dealt with and
        # store the fragments in the list defined below
        current_seq_frags = []

        start_index = 0
        end_index = pure_seq_len

        for _ in range(n_fragments):
            current_fragment = nt_seq[start_index:end_index]

            # Increase the indices in order to obtain the next fragment
            # in the next iteration step
            start_index += pure_seq_len
            end_index += pure_seq_len

            # Decide whether to use or to discard the current fragment
            if len(current_fragment) >= terminal_cutoff:
                current_seq_frags.append(current_fragment)
            # Taking into account the case that one complete coding
            # sequence is shorter than the desired fragment length
            elif n_fragments == 1:
                current_seq_frags.append(current_fragment)
            else:
                continue

        # Append the gathered fragments of the sequence currently dealt
        # with to the global fragment list
        frags_list += current_seq_frags

    return frags_list


def _tokenise_nt_seq(seq):
    """
    This function is analogous to the function `_tokenise_aa_seq` and
    performs tokenisation for nucleotide sequences. However, instead of
    taking the individual nucleotide letters as tokens and mapping them
    to integer indices, three successive nucleotides, i. e. triplets /
    codons are taken as tokens. Hence, there are 64 unique tokens,
    without the additional tokens (e. g. <START>, <END>, etc.).

    For this purpose, it employs the private function
    `_split_into_triplets`.

    Parameters
    ----------
    seq: str or binary
        The input nucleotide sequence to be tokenised.

    Returns
    -------
    tokenised_seq: list, dtype=int
        The tokenised analogue of the input nucleotide sequence.
    """

    # It is important to prepend the <START> token and append the <END>
    # token to the sequence
    tokenised_seq = (
            [additional_codon_token_to_index["<START>"]]
            +
            [
                codon_token_to_index.get(codon) for codon
                in _split_into_triplets(_parse_seq(seq))
            ]
            +
            [additional_codon_token_to_index["<END>"]]
    )

    return tokenised_seq


def _tokenise_nt_seqs(seqs):
    """
    This function performs tokenisation for a whole data set of
    nucleotide sequences.

    For this purpose, it employs the private function
    `_tokenise_nt_seq`.

    Parameters
    ----------
    seqs: iterable, dtype=str or dtype=binary
        An iterable containing the nucleotide sequences to be tokenised.
    seq_len: int
        Integer indicating the maximum sequence length, but in terms of
        codons / triplets instead of in terms of single nucleotides.
        Input sequences shorter than this value are padded with a
        special token until they reach the specified length. When
        passing this value to the function call, it should be taken into
        consideration that the <START> and <END> tokens are already
        added to each individual sequence. Hence, in order to avoid a
        ValueError, the sequence length needs to be incremented by 2.

        Apart from that, the tokenised nucleotide sequence is shortened
        by one unit at the beginning and the end later on, respectively.
        Hence, tokenised_seq = _tokenise_aa_seq(seq)
        if tokenised_seq is None:
            print(f"Warning: Tokenisation returned None for sequence {seq}")
            tokenised_seq = []  # Handle or skip as appropriateits length must be greater by 1 than the length of the
        corresponding amino acid sequence. This, however, is handled by
        the function itself, so that the user can insert the same
        sequence length for both the amino acid and the nucleotide
        sequence.

    Returns
    -------
    tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
        A NumPy array harbouring  the tokenised analogues of the input
        nucleotide sequences. The first dimension, m, corresponds to the
        amount of input sequences, whereas the second dimension, n,
        corresponds to the largest sequence length of the batch.
    """
    # First, perform the mere tokenisation
    tokenised_seqs_list = []

    for seq in seqs:
        tokenised_seq = _tokenise_aa_seq(seq)
        if tokenised_seq is None:
            print(f"Warning: Tokenisation returned None for sequence {seq}")
            tokenised_seq = []  # 빈 리스트로 처리
        tokenised_seqs_list.append(_tokenise_nt_seq(seq))

    # Perform padding for each of the tokenised sequences
    # Determine the largest sequence length in the batch and increment it by 1
    max_length = max(map(len, tokenised_seqs_list))
    max_length += 1

    tokenised_seqs_array = np.array(
        [tokenised_seq
         +
         (
                 max_length - len(tokenised_seq)
         ) * [additional_codon_token_to_index['<PAD>']]
         for tokenised_seq in tokenised_seqs_list],
        dtype=np.int32
    )

    return tokenised_seqs_array


def tokenise_nt_seqs_for_eval(seqs, max_seq_len):
    """
    This function performs tokenisation for a whole data set of
    nucleotide sequences. For this purpose, it employs the private
    function '_tokenise_nt_seq'.

    The difference between this function and '_tokenise_nt_seqs' is that
    this function has an additional parameter, 'max_seq_len', indicating
    the maximum sequence length in terms of codons/triplets. This is
    opposed to the behaviour of the function '_tokenise_nt_seqs', which
    adopts the length of the largest input sequence as the maximum
    sequence length.

    Due to this implementation, 'max_seq_len' must be at least equal to
    the length of the largest input sequence. Otherwise, a ValueError is
    raised.

    Parameters
    ----------
    seqs: iterable, dtype=str or dtype=binary
        An iterable containing the nucleotide sequences to be tokenised.
    max_seq_len: int
        An integer indicating the maximum sequence length, but in terms
        of codons/triplets instead of in terms of single nucleotides.
        Input sequences shorter than this value are padded with a
        special token until they reach the specified length. When
        passing this value to the function call, it should be taken into
        consideration that the <START> and <END> tokens are already
        added to each individual sequence. Hence, in order to avoid a
        ValueError, the sequence length needs to be incremented by 2.

        Apart from that, the argument passed to this parameter needs to
        be at least equal to the length of the largest input sequence.
        Otherwise, a ValueError is raised.

    Returns
    -------
    tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
        A NumPy array harbouring the tokenised analogues of the input
        nucleotide sequences. The first dimension, m, corresponds to the
        amount of input sequences, whereas the second dimension, n,
        corresponds to the maximum sequence length.
    """
    # First, check for a valid argument passed to 'max_seq_len'
    # Keep in mind that the parameter 'max_seq_len' refers to the
    # maximum sequence length in terms of codons, not in terms of single nucleotides
    # Hence, in order to perform the comparison, the length of the
    # largest input nucleotide sequence must be divided by three
    max_input_length = max(map(len, seqs)) / 3
    if max_seq_len < max_input_length:
        raise ValueError(
            "The argument passed to 'max_seq_len' must be at least "
            "equal to the length of the largest input sequence!"
        )

    tokenised_seqs_list = []

    for seq in seqs:
        tokenised_seqs_list.append(_tokenise_nt_seq(seq))

    # Perform padding for each of the tokenised sequences
    tokenised_seqs_array = np.array(
        [tokenised_seq
         +
         (
                 max_seq_len - len(tokenised_seq)
         ) * [additional_codon_token_to_index['<PAD>']]
         for tokenised_seq in tokenised_seqs_list],
        dtype=np.int32
    )

    return tokenised_seqs_array


def _detokenise_nt_seqs(nt_token_tensor):
    """
    This function performs detokenisation for a whole data set of
    nucleotide sequences.

    Parameters
    ----------
    nt_token_tensor: pass
        pass

    Returns
    -------
    nt_seq_array: pass
        pass
    """
    # Convert the TensorFlow tensor into a NumPy array as they are much
    # more convenient to handle
    nt_token_array = nt_token_tensor.numpy()

    nt_seq_list = []

    for token_seq in nt_token_array:
        codon_list = [
            index_to_codon_token.get(token)
            for token in token_seq
        ]
        codon_str = "".join(codon_list)
        nt_seq_list.append(codon_str)

    # Convert the list into a NumPy array
    nt_seq_array = np.array(nt_seq_list, dtype=str)

    return nt_seq_array


def format_dataset(aa_nt_pairs):
    """
    This function formats the data set into a shape digestible by the
    transformer architecture. To be more precise, a tuple `(inputs,
    targets)` is returned.

    `inputs` is a tuple containing two elements, `encoder_inputs` and
    `decoder_inputs`. `encoder_input` is the tokenised amino acid
    sequence, while `decoder_inputs` is the tokenised nucleotide
    sequence "so far", i. e. the codons 0 to N used to predict
    nucleotide token N+1 and beyond in the target nucleotide sequence.

    `targets` is the target nucleotide sequence offset by one step.
    Hence, it provides the next codons in the target nucleotide
    sequence, i. e. what the transformer will try to predict.

    For this purpose, it employs the private functions `_encode_aa_seqs`
    as well as `_tokenise_nt_seqs`.

    Parameters
    ----------
    aa_nt_pairs: Pandas DataFrame, shape=(n, 2)
        The Pandas DataFrame containing either the training or the
        validation data set. It consists of two columns / series, of
        which the first represents the amino acid sequences and the
        second one represents the corresponding nucleotide sequences.

    Returns
    -------
    output_tuple: tuple, shape=(2,)
        Tuple containing the tokenised input and target necessary for
        training and validation of the sequence-to-sequence transformer.
        The first tuple element is another tuple comprising two arrays,
        corresponding to the encoder input (tokenised amino acid
        sequence) and the decoder input (tokenised nucleotide sequence)
        of the transformer. The second tuple element is the target
        nucleotide sequence offset by one step.
    """
    # As a first step, convert the Pandas DataFrame into a NumPy array
    # as the latter is easier to handle with respect to iteration, etc.
    aa_nt_pairs = aa_nt_pairs.to_numpy()

    aa_seqs, nt_seqs = zip(*aa_nt_pairs)
    # The zip function generates an iterator of tuples; convert them to
    # lists
    aa_seqs = list(aa_seqs)
    nt_seqs = list(nt_seqs)

    # Now, perform tokenisation by employing the respective private
    # function
    aa_encoding = _tokenise_aa_seqs(aa_seqs)
    nt_encoding = _tokenise_nt_seqs(nt_seqs)

    # Now, create the output tuple
    # Keep in mind that the decoder input is shortened by one unit at
    # the end, whereas the target nucleotide sequence is shortened by
    # one unit at the beginning
    # This is done so that the transformer does not know the answer in
    # advance, but is trained to predict the next token
    # Compared to the transformer involving ProteinBERT, not a
    # dictionary is returned, but two separate objects, of which the
    # first is a tuple
    # The reason for this change is that the input layers of the encoder
    # and the decoder do not carry any names any more
    # Instead, the order in which the subelements of the first returned
    # element (tuple) are arranged must be paid attention to, as the
    # transformer implementation dealt with assumes the first subelement
    # to be the so-called "context", i. e. the source sentence, and the
    # second subelement to be the target sentence
    return (aa_encoding, nt_encoding[:, :-1]), nt_encoding[:, 1:]


def _determine_actual_seq_lens(encoding_array):
    """
    This private function accepts as input a two-dimensional NumPy array
    containing tokenised sequences (integer token sequences) and returns
    a one-dimensional tensor containing the actual sequence lengths, i.
    e. the sequence lengths without padding characters.

    Conveniently, the padding integer is zero in all sequences, allowing
    the usage of `tf.math.count_nonzero`.

    As a last step, the type of the obtained tensor (int) is converted
    to float32, as the element-wise multiplication of two tensors
    requires them to have the same data type.

    Parameters
    ----------
    encoding_array: NumPy array, shape=(m, n), dtype=int
        A two-dimensional NumPy array harbouring the tokenised sequences
        as integer token sequences. The first dimension, m, represents
        the individual sequences, whereas the second dimension, n,
        represents the individual sequence positions.

    Returns
    -------
    actual_seq_lens: tensor, shape=(m,), dtype=float
        A one-dimensional TensorFlow tensor harbouring the actual length
        of the individual sequences, i. e. the sequence length excluding
        padding characters (zero).
    """
    actual_seq_lens_int_type = tf.math.count_nonzero(
        input=encoding_array,
        axis=1
    )

    actual_seq_lens = tf.cast(
        x=actual_seq_lens_int_type,
        dtype=tf.float32
    )

    return actual_seq_lens


def translate_aa_into_nt(
        transformer, aa_seqs, max_seq_length, return_string=True, batch_size=32
):
    """
    This function translates from the "language" of amino acids into one
    specific "dialect" of the nucleotide sequence "language". The term
    "dialect" denotes the codon usage bias of one specific organism,
    e. g. E. coli or S. cerevisiae. In order to accomplish this task,
    the function accepts as input a transformer neural network which was
    trained to translate amino acid sequences into nucleotide sequences
    of one specific organism.

    For this purpose, it employs the private functions `_encode_aa_seqs`
    as well as `_detokenise_nt_seqs`.

    In order to circumvent an OOM error in case of very large test data
    sets, the test data set is manually batched into batches of a
    user-defined size; the default value is 32.

    Parameters
    ----------
    transformer: Keras model
        A transformer neural network implemented in Keras and performing
        the translation of amino acid sequences into nucleotide
        sequences according to the codon usage bias of one specific
        organism.
    aa_seqs: list
        A list comprising the amino acid sequences to be translated into
        nucleotide sequences in accordance with the codon usage bias of
        the respective organism.
    max_seq_length: int
        An integer denoting the maximum sequence length after which to
        stop translation in case that the <END> token is not produced.
    return_string: boolean, optional
        A boolean indicating whether the translations are supposed to be
        returned as contiguous strings (i. e. one string for each input
        amino acid sequence) or as integer tokens. Defaults to 'True'.
    batch_size: int
        An integer defining the batch size. The segmentation of the data
        set into batches of a defined size happens in order to avoid an
        OOM error.

    Returns
    -------
    nt_translations: NumPy array, dtype=str or NumPy array, dtype=int
        Depending on the argument of the optional parameter
        'return_string', two different outputs can be returned. If the
        optional parameter is set to 'True', a one-dimensional NumPy
        array is returned harbouring the sequences generated during
        translation as strings. In other words, it contains the
        generated sequences as human-readable nucleotide sequences. Its
        length equals the amount of amino acid sequences passed as
        input, i. e. the length of `aa_seqs`. However, if the optional
        parameter is set to 'False', a two-dimensional NumPy array is
        returned harbouring the sequences generated during translation
        as integer tokens. Each row of this two-dimensional array
        corresponds to one translation.
    """
    # Tokenise the input amino acid sequences
    aa_encoding = _tokenise_aa_seqs(aa_seqs)

    # Determine the actual sequence lengths via the private function
    actual_seq_lens = _determine_actual_seq_lens(aa_encoding)

    # The manual batching of the tokenised amino acid sequences is
    # performed
    # To this end, the total amount of batches having the size defined
    # by the user is determined
    n_batches = ceil(len(aa_encoding) / batch_size)

    # As stated in the documentation string, the output depends on the
    # value of the optional parameter 'return_string'
    # Hence, depending on whether the parameter is set to 'True' or
    # 'False', a different output array is created
    if return_string:
        nt_translations = np.array([], dtype=str)
    else:
        nt_translations = np.empty(
            shape=(0, max_seq_length), dtype=np.int32
        )

    for i in range(n_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size

        # Extract the batch currently dealt with
        current_batch_aa_encoding = aa_encoding[start_index:end_index, :]
        # Do not forget to extract the respective portion of the array
        # harbouring the actual sequence lengths!
        current_actual_seq_lens = actual_seq_lens[start_index:end_index]

        # Define a function that computes the token probabilities for the
        # next position given the input sequence
        # The transformer output has the shape (B, l, 68), where B denotes
        # the batch size/amount of amino acid sequences entered as input,
        # l denotes the maximum sequence length and 68 is the total amount
        # of nucleotide tokens
        # As only the probabilities of the next token to predict are wanted,
        # "-1" is chosen in the second dimension during slicing

    def token_probability_fn(decoder_input_tokens):

        return transformer(
            # The keyword argument `training` is set to `False` as this
            # makes inference more efficient by only computing the last
            # the last prediction instead of the prediction for all tokens
            # It is not necessary to distinguish between training and inference as this function is exclusively used for
            # inference; hence, this keyword argument does not need to be accessible to the user
            [current_batch_aa_encoding, decoder_input_tokens, current_actual_seq_lens],
            training=False
        )[:, -1, :]

    n_seqs = len(current_batch_aa_encoding)

    # Initialise the translated nucleotide sequences with the "<START>"
    # token
    prompt = tf.fill(
        (n_seqs, 1),
        global_codon_token_to_index["<START>"]
    )


# me
def greedy_search(transformer, start_sequence, max_length, vocab_size):
    sequence = start_sequence  # Initialize the sequence with the start sequence

    for _ in range(max_length):
        # Generate predictions for the current sequence
        predictions = transformer.predict(sequence, verbose=0)

        # Select the next word with the highest probability
        next_word = np.argmax(predictions[0])

        # Append the next word to the sequence
        sequence = np.append(sequence, next_word)

        # Remove the first word to maintain the sequence length
        sequence = sequence[1:]

        # Reshape the sequence to a 2D tensor
        sequence = np.expand_dims(sequence, axis=0)

    return sequence

    # Example start sequence (e.g., [1, 2, 3, 4])
    start_sequence = np.random.randint(0, vocab_size, (1, max_sequence_length))

    # Maximum length of the generated sequence
    # max_length = 10

    # Generate a sequence using greedy search
    generated_sequence = greedy_search(transformer, start_sequence, max_length, vocab_size)
    print("Generated sequence:", generated_sequence)

    # Strangely enough, it is stated in the Keras documentation that
    # 'keras_nlp.utils.greedy_search' returns either a 1D int Tensor
    # or a 2D int RaggedTensor, whereby a ragged tensor is a
    # non-rectangular tensor whose entries, i. e. rows have
    # different sizes
    # However, this is not true as the chosen padding token is added
    # after encountering the end token until the maximum sequence
    # length is reached
    generated_tokens = greedy_search(
        token_probability_fn,
        prompt,
        max_length=max_seq_length,
        end_token_id=global_codon_token_to_index["<END>"]
    )

    if return_string:
        current_batch_nt_translations = _detokenise_nt_seqs(generated_tokens)
        nt_translations = np.append(nt_translations, current_batch_nt_translations)
    else:
        current_batch_nt_translations = generated_tokens.numpy()
        # Verify that the array harbouring the translations is two-
        # dimensional; an one-dimensional array is returned when the
        # batch size is 1, which can occur under certain
        # circumstances
        if current_batch_nt_translations.ndim < 2:
            current_batch_nt_translations = np.expand_dims(
                current_batch_nt_translations, axis=0
            )
        # Now, append the new translations to the output array
        nt_translations = np.append(
            nt_translations, current_batch_nt_translations, axis=0
        )

    return nt_translations
