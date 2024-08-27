import pickle

import tensorflow as tf
from tensorflow import keras

from transformer_wip import Transformer
from auxiliary_file import train_validate_test_split
from tokenisation import format_dataset, \
    N_AA_TOKENS, N_CODON_TOKENS

# The constant `MAX_SEQ_LEN` must be adjusted to the data set currently dealt with!
# In the case of the E. coli data set comprising 6 genomes, the maximum sequence length was set to 500
# Hence, including a safety buffer, the constant is set to 1000
MAX_SEQ_LENGTH = 1000

# Perform the train-validate-test split
# With respect to paths, it is important to keep in mind that the dollar
# sign must not be used within Python scripts in order to denote a variable
# This is due to the fact that the dollar sign is only recognised by the
# shell/bash as designating a variable
# Hence, instead of using the environment variable $HPC_SCRATCH, the
# full path must be written, i.e. /work/scratch/ja63nuwu
training_set, validation_set, test_set = train_validate_test_split(
    path_to_csv=f"/work/home/yj90zihi/Bachelor_Yurim/Transformer/csv_files/e_coli_aa_nt_seqs.csv",
    train_size=0.8,
    valid_size=0.1
)
print("Data information was successfully retrieved!")

# Define the hyperparameters of the transformer
# They are intentionally chosen to be smaller than the ones from the
# base model described in the publication ("Attention is all you need")
# This is due to the fact that merely a proof of concept is supposed to be achieved
num_layers = 3
d_model = 128
dff = 512
num_heads = 4
dropout_rate = 0.1

# The command below releases, i.e. clears the global state managed by
# Keras
keras.backend.clear_session()

# Instantiate the Transformer model
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
print("The transformer was successfully built!")


# Set up the optimiser
# The Adam optimiser is employed in conjunction with a custom learning rate scheduler
# The learning rate scheduler follows the formula in the original Transformer paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


# Set up the loss and metrics
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


# The training procedure is configured by using the `compile` method
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

# Define callbacks in order to be able to control the training process
training_callbacks = [
    # With respect to the EarlyStopping callback, it must be
    # mentioned that the keyword argument `start_from_epoch` is
    # recognised only from TensorFlow version 2.11.0 onward
    keras.callbacks.EarlyStopping(
        monitor="masked_accuracy",
        min_delta=0.005,
        patience=5,
        verbose=1,
        mode="max",
        restore_best_weights=True
    ),
    # Add a checkpoint callback in order to save the model weights periodically
    # This way, in case of an interruption, training does not need to be
    # repeated from the very beginning, but can be continued from the
    # last saved state
    keras.callbacks.ModelCheckpoint(
        filepath="Checkpoint_directory/best_weights.weights.h5",
        monitor="masked_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        save_freq="epoch"
    )
]

# Format the training and validation data set
transformer_inputs, transformer_outputs = format_dataset(training_set)
formatted_validation_ds = format_dataset(validation_set)
print("Formatting of the data set was successful!")

# Finally, perform the training via the `fit` method
transformer.fit(
    x=transformer_inputs,
    y=transformer_outputs,
    batch_size=32,
    epochs=50,
    callbacks=training_callbacks,
    validation_data=formatted_validation_ds
)
print("Training was successful!")

# After completion of the training, the weights are pickled and saved to disk
# Apart from that, the specifications of the transformer architecture are saved as well
transformer_weights = transformer.get_weights()
architecture_specifications = (
    num_layers, d_model, dff, num_heads, dropout_rate
)

with open("E-coli-6-genomes-trained-weights_6th_approach_window_width_101.pkl", "wb") as f:
    pickle.dump(
        [transformer_weights, architecture_specifications], f
    )
print(
    "The weights of the trained transformer as well as its transformer "
    "specifications have been successfully saved to disk!"
)
