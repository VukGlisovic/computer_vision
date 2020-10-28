import tensorflow as tf
from crnn.constants import *


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(DIR_MODELS, RUN_NR, 'checkpoints', 'crnn_ep{epoch:03d}.h5'),
    monitor='val_normalized_edit_distance',
    save_best_only=True,
    save_weights_only=True,
    mode='max'
)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=2,
    verbose=1,
    mode='min',
    min_delta=0.001,
    cooldown=0,
    min_lr=1e-7
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(DIR_MODELS, RUN_NR, 'tensorboard'),
    histogram_freq=0,
    write_graph=True,
    write_grads=False,
    write_images=False,
    profile_batch='30,40',
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_normalized_edit_distance',
    min_delta=0.001,
    patience=7,
    verbose=0,
    mode='max',
    restore_best_weights=False
)
