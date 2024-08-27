import os
import csv
import shutil
import pandas as pd
import Levenshtein
from sklearn.model_selection import train_test_split
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta



def create_csv_file(path_to_genomes, host_name, max_genome_num=None, max_seq_len=None):
    """
    This function generates a CSV file comprising all the genomes used
    for training, validating and testing the transformer.
    
    As opposed to the Chinese publication by Tian et al. (Nature, 2017),
    the test data set is not confined to the sequence E. coli
    K12_MG1655. Instead, the whole data set is shuffled and subsequently
    subjected to the train-validate-test split.
    
    Apart from that, it must be mentioned that the Chinese authors
    erroneously listed four genomes twice in the supplementary material
    (not only the strain description, but also the unique identifier
    UID). Hence, instead of 64 genomes, only 60 genomes served as the
    training data set. Furthermore, the complete genome of Escherichia
    coli str. K-12 substr. W3110 (ASM1024v1) can not be opened after
    downloading it. For this reason, the genome ASM187869v1 (chromosome)
    is employed as a substitute.

    The CSV file contains two columns, of which the first represents
    amino acid sequences of coding genes and the second represents the
    respective nucleotide sequence. This is comparable to training a
    transformer to translate e.g. from English to French, in which case
    the first column of the CSV file would contain English sentences and
    the second column would comprise their French counterparts.

    Additionally, a second file is generated, which is a text file
    containing information regarding the sequences in the CSV file, such
    as the total amount of sequences meeting the prerequisites (only
    unambiguous alphabet, sequence length is a multiple of 3).

    A third file is generated, too. This third file is a text file
    listing all the genomes incorporated in the respective CSV file.
    This information is necessary as in case that not all genomes in the
    specified directory are employed, but only a subset (by using the
    optional `max_seq_len` parameter in the function call), the chosen
    genomes are not necessarily identical with the first `max_seq_len`
    genomes in the respective directory. This is due to the fact that
    `os.listdir()` lists the entries in the respective directory in an
    arbitrary order.

    Parameters
    ----------
    path_to_genomes: str
        A string denoting the path to the directory containing the
        genomes to be processed.
    host_name: str
        The name of the expression host for which the CSV file is
        supposed to be created.
    max_genome_num: int, optional
        The maximum amount of genomes to be incorporated in the CSV
        file. If this optional parameter is not specified in the
        function call, all genomes in the specified directory are
        incorporated.
    max_seq_len: int, optional
        The maximum length sequences to be incorporated are allowed to
        have. This optional parameter is important with respect to
        memory restrictions as ordinary attention grows quadratically in
        sequence length.
    """
    genome_files_list = os.listdir(path_to_genomes)

    total_amount_of_genomes = len(genome_files_list)
    if max_genome_num is None:
        max_genome_num = total_amount_of_genomes

    incorporated_genomes_list = []

    CDS_counter = 0
    max_nt_seq_len = 0

    # The "newline" parameter below must be specified as an empty string
    # in order to prevent subsequent lines from being separated by an empty line
    with open(f"{host_name}_aa_nt_seqs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Amino acid sequence", "Nucleotide sequence"]
        writer.writerow(header)
        file_paths = []

        for directory_path, directory_names, filenames in os.walk(path_to_genomes):
            for filename in filenames:
                # Construct the full file path
                full_path = os.path.join(directory_path, filename)
                # Add the file path to the list
                file_paths.append(full_path)

        # Iterate through the entire directory, translate each
        # nucleotide sequence into the respective amino acid sequence
        # and add the row to the CSV file
        for i, filename in enumerate(genome_files_list):
            if i == max_genome_num:
                break

            incorporated_genomes_list.append(filename + "\n")

            file_path = os.path.join(path_to_genomes, filename)
            for _, nt_seq in fasta.FastaFile.read_iter(file_path):
                # A translation of the complete nucleotide sequence
                # requires the sequence length to be a multiple of 3,
                # which is why sequences not meeting this requirement
                # are discarded
                current_nt_seq_len = len(nt_seq)
                if (current_nt_seq_len % 3) != 0:
                    continue
                # Skip sequences whose length exceeds the user-defined maximum
                # As one token is later generated for three nucleotides
                # / one codon, the amount of amino acid residues and the amount of tokens are the same
                if max_seq_len is not None:
                    current_aa_seq_len = current_nt_seq_len // 3
                    if current_aa_seq_len > max_seq_len:
                        continue
                # Furthermore, in order for the translation process to
                # successfully take place, the sequences may exclusively
                # contain the letters of the unambiguous alphabet, i.e. the letters A, T, G and C
                # All sequences containing letters of the ambiguous alphabet are therefore discarded
                if "R" in nt_seq:
                    continue
                if "Y" in nt_seq:
                    continue
                if "W" in nt_seq:
                    continue
                if "S" in nt_seq:
                    continue
                if "M" in nt_seq:
                    continue
                if "K" in nt_seq:
                    continue
                if "H" in nt_seq:
                    continue
                if "B" in nt_seq:
                    continue
                if "V" in nt_seq:
                    continue
                if "D" in nt_seq:
                    continue
                if "N" in nt_seq:
                    continue

                # The sequence currently dealt with meets the
                # prerequisites for being processed, which is why the
                # counter for coding sequences is incremented by 1
                CDS_counter += 1

                # Evaluate whether the sequence currently dealt with
                # has the largest length so far
                if current_nt_seq_len > max_nt_seq_len:
                    max_nt_seq_len = current_nt_seq_len

                # Presumably due to sequencing inaccuracies, the first
                # nucleotide sometimes is different from A, although the
                # second and third ones are T and G, respectively
                # Hence, in order to restore the start codon ATG, the
                # single nucleotide exchange is performed
                if nt_seq[0] != "A":
                    nt_seq = "A" + nt_seq[1:]

                # Now, perform the translation into the amino acid
                # sequence and add the entry to the CSV file
                aa_seq = seq.NucleotideSequence(nt_seq).translate(
                    complete=True
                )
                writer.writerow([aa_seq, nt_seq])

    # The text file containing information regarding the sequences in
    # the CSV file is generated
    # Logically, the maximum amino acid sequence length equals the
    # maximum nucleotide sequence length divided by three, as one codon
    # comprising three nucleotides encodes one amino acid
    max_aa_seq_len = int(max_nt_seq_len / 3)

    lines = [
        f"The maximum sequence length was set to {max_seq_len}.\n"
        "The total amount of coding sequences meeting the prerequisites"
        f" is {CDS_counter}.\n",
        f"The largest nucleotide sequence encompasses {max_nt_seq_len} "
        "nt.\n",
        "Accordingly, the largest amino acid sequence encompasses "
        f"{max_aa_seq_len} amino acids.\n",
        "Keep in mind that the number of tokens, however, is the same, "
        "irrespective of whether nucleotides or amino acids are dealt\n"
        "with, as one codon and one amino acid are considered as one "
        "token, respectively!"
    ]

    with open(f"Information_on_{host_name}_csv_file.txt", "w") as f:
        f.writelines(lines)

    with open(
            f"Incorporated_sequences_in_{host_name}_csv_file.txt", "w"
    ) as f:
        f.write(
            f"The following {max_genome_num} genomes were incorporated in "
            "the CSV file:\n"
        )
        f.writelines(incorporated_genomes_list)


def train_validate_test_split(path_to_csv, train_size, valid_size):
    """
    This function reads in a CSV file and performs a train-validate-test
    split on the data by applying scikit-learn's train_test_split twice
    in succession.
    """
    aa_nt_df = pd.read_csv(path_to_csv)

    training_set, remaining_set = train_test_split(
        aa_nt_df,
        train_size=train_size,
        random_state=0
    )

    # It must be kept in mind that the basic value/entity the validation
    # set size is referring to now is different from the whole CSV file
    # Hence, this relative quantity must be adjusted
    valid_size = valid_size / (1 - train_size)
    validation_set, test_set = train_test_split(
        remaining_set,
        train_size=valid_size,
        random_state=0
    )

    return training_set, validation_set, test_set


def compute_norm_levenshtein(str_1, str_2):
    """
    This function computes the normalised Levenshtein distance between
    two strings, which is beneficial for the comparison of sequences
    that are of unequal length.

    Parameters
    ----------
    str_1: string
        The first of the two strings to perform the distance computation
        on.
    str_2: string
        The second of the two strings to perform the distance
        computation on.

    Returns
    -------
    normalised_levenshtein: float
        The normalised Levenshtein distance
    """
    unnormalised_levenshtein = Levenshtein.distance(str_1, str_2)
    max_length = max(len(str_1), len(str_2))
    normalised_levenshtein = (
            (max_length - unnormalised_levenshtein) / max_length
    )

    return normalised_levenshtein


def _count_codons(nt_seq):
    """
    This function counts the amount of times individual codons appear
    within a given nucleotide sequence. To this end, the sequence is
    split into codons/triplets. Hence, the sequence's length must be a
    multiple of three.

    Parameters
    ----------
    nt_seq: str
        The nucleotide sequence for which the codon counts are to be
        determined.

    Returns
    -------
    codon_count: dict
        A dictionary listing the number of occurrences of individual
        codons. Accordingly, its keys are codons (strings), while its
        values are integers.
    """
    n_codons = len(nt_seq) // 3

    codon_count = {}

    for i in range(n_codons):
        start_index = i * 3
        end_index = (i + 1) * 3

        current_codon = nt_seq[start_index:end_index]
        codon_count[current_codon] = codon_count.get(current_codon, 0) + 1

    return codon_count


def convert_codon_freqs_to_RNA(DNA_dict):
    """
    This function accepts as input a dictionary containing as keys DNA
    codons and as values the respective codon usage frequencies in per
    mille. The dictionary keys are converted to their RNA analogue,
    i. e. T denoting thymine is replaced with U denoting uracil.

    It must be kept in mind that the codons on the coding strand and the
    respective mRNA molecule are identical except for the fact that T is
    replaced by U. This is due to the fact that not the coding strand,
    but the non-coding strand serves as template for the generation of
    the mRNA molecule.

    Parameters
    ----------
    DNA_dict: dictionary
        A dictionary containing as keys DNA codons and as values the
        respective codon usage frequencies.

    Returns
    -------
    RNA_dict: dictionary
        A dictionary containing as keys the RNA analogue of the keys of
        the input dictionary.
    """
    RNA_dict = {}

    for DNA_codon, freq in DNA_dict.items():
        # Convert the DNA codon to the respective RNA codon by replacing
        # T with U
        RNA_codon = DNA_codon.replace("T", "U")

        # Add the entry to the output dictionary
        RNA_dict[RNA_codon] = freq

    return RNA_dict


path_to_E_coli_genomes = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\datasets\E.coli_Jacob_short'
path_to_B_subtilis_genomes = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\datasets\B.subtilis_Jacob_short'
path_to_Corynebacterium_glutamicum_genomes = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\datasets\C_glutamicum_Jacob\ncbi_dataset\data'

base_folder_path = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\datasets\E_coli_Jacob\data'


# Generate a CSV file for Escherichia coli genomes
#if __name__ == "__main__":
#   create_csv_file(
#       path_to_genomes=destination_folder,
#      host_name="e_coli",
#     max_genome_num=6,
#    max_seq_len=500
#)


# Generate a CSV file for Bacillus subtilis genomes
#if __name__ == "__main__":
#create_csv_file(
#path_to_genomes=path_to_B_subtilis_genomes,
#host_name="B_subtilis_6_genomes",
#max_genome_num=6,
#max_seq_len=500
#)

# Generate a CSV file for Corynebacterium glutamicum genomes
#if __name__ == "__main__":
#create_csv_file(
#path_to_genomes=path_to_Corynebacterium_glutamicum_genomes,
#host_name="Corynebacterium_glutamicum_6_genomes",
#max_genome_num=6,
#max_seq_len=500
#)
def move_fna_files(source_folder, destination_folder):
    """
    Moves all .fna files from subdirectories of the source_folder to the destination_folder,
    renaming files to include their original directory names to avoid name conflicts.

    Parameters:
    - source_folder: str, path to the root folder containing subdirectories with .fna files.
    - destination_folder: str, path to the folder where .fna files should be moved.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Destination folder created: {destination_folder}")

    # Variable to keep track of the number of files moved
    file_moved_count = 0

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file has a .fna extension
            if file.endswith(".fna"):
                # Construct full file path
                file_path = os.path.join(root, file)

                # Generate a unique filename by including the folder name
                folder_name = os.path.basename(root)
                new_file_name = f"{folder_name}"
                dest_file_path = os.path.join(destination_folder, new_file_name)

                # Move file to the destination folder with a unique name
                shutil.move(file_path, dest_file_path)
                print(f"Moved: {file_path} -> {dest_file_path}")

                # Increment the count of moved files
                file_moved_count += 1

    print(f"Total files moved: {file_moved_count}")


# Example usage
source_folder = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\datasets\C_glutamicum_Jacob'
destination_folder = r'C:\Users\dbfla\PycharmProjects\Bachelor_Yurim\Transformer\collected_fna_files_C.glutamicum_Jacob'

move_fna_files(source_folder, destination_folder)

if __name__ == "__main__":
    create_csv_file(
        path_to_genomes=destination_folder,
        host_name="C_glutamicum",
        max_genome_num=6,
        max_seq_len=500
    )