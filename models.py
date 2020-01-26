# The file that contains hand-crafted models.

import tensorflow as tf
import tensorflow.keras as keras

'''
The number of unique words we want to remember in the model.
'''
VOCAB_SIZE = 1000

def define_and_compile_models():
    '''
    Defines many possible combinations for the models, can easily be extended.
    '''
    optimizers = ['adam', 'rmsprop']
    activations = ['relu', 'tanh', 'softmax']
    rnn_layers = [1, 2]

    # TODO: Try with different RNN cells, change the number of neurons in the hidden layer, change learning rates or add decay.

    models = []

    for o in optimizers:
        for a in activations:
            for num in rnn_layers:
                if num == 2:
                    model = tf.keras.Sequential([
                        tf.keras.layers.Embedding(VOCAB_SIZE, 16),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
                        tf.keras.layers.Dense(4, activation=a),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                if num == 1:
                    model = tf.keras.Sequential([
                        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, recurrent_dropout=0.2)),
                        tf.keras.layers.Dense(4, activation=a),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
                model.compile(optimizer=o, loss='binary_crossentropy', metrics=metrics)
                models.append(model)

    return models


def define_and_compile_model1():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 16),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model2():
    '''
    The preferred model, embeddings results are generated by using this model.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, recurrent_dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dense(2, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model3():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model4():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, recurrent_dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model5():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model6():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model7():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, recurrent_dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model8():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, recurrent_dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model9():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dense(2, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model10(): 
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)),
        tf.keras.layers.Dense(2, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model11():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    return model

def define_and_compile_model12():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 8),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6)),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    metrics = ['acc',keras.metrics.Precision(), keras.metrics.Recall()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    return model