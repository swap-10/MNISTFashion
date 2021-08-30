import tensorflow as tf


def defmodel():
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(10)
                                 ])
    return model


def compilemodel(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
