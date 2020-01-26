import random
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from models import define_and_compile_model2, VOCAB_SIZE
from embedding_generator import generate_embeddings


'''
The maximum review length.
'''
MAX_LENGTH = 30

class CustomEarlyStopping(EarlyStopping):
    '''
    Defines conditions for early stop of the training process.
    '''
    def __init__(self, val_acc_th, acc_th, min_epochs, **kwargs):
        super(CustomEarlyStopping, self).__init__(**kwargs)
        self.validation_accuracy_threshold = val_acc_th
        self.accuracy_threshold = acc_th
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Get validation accuracy.
        current_validation_accuracy_threshold = logs.get('val_acc')
        current_accuracy_threshold = logs.get('acc')

        # Stop only if both training and validation accuracy are bigger than some threshold (which brings low loss)
        if (epoch >= self.min_epochs) & (current_validation_accuracy_threshold >= self.validation_accuracy_threshold) & (current_accuracy_threshold > self.accuracy_threshold):
            self.stopped_epoch = epoch
            self.model.stop_training = True



def test_models(positive_data, negative_data, training_split):
    '''
    The helper method used to measure accuracy, precision and recall of the models.
    '''
    res = []

    # Enter models to test.
    models = [
        define_and_compile_model2
    ]

    for model in models:
        # Change seed for each test.
        res.append(test_one_model(positive_data, negative_data, training_split, model, 37))
        res.append(test_one_model(positive_data, negative_data, training_split, model, 58))
        res.append(test_one_model(positive_data, negative_data, training_split, model, 1003))
    
    print (res)

def test_one_model(positive_data, negative_data, training_split, fetch_model, seed_num):
    '''
    Executes Monte Carlo validation (very similar to k-fold validation if with non-random seed).
    '''
    acc_arr = []
    prec_arr = []
    rec_arr = []

    positive_out, negative_out = self.create_outputs(len(positive_data))

    random.seed(seed_num)

    # Monte Carlo validation for 10 times.
    for i in range(10):
        random.shuffle(positive_data)
        random.shuffle(negative_data)

        _, acc, prec, recall = train_model(positive_data + negative_data, np.concatenate((positive_out, negative_out), axis=None), training_split, fetch_model)
        acc_arr.append(acc)
        prec_arr.append(prec)
        rec_arr.append(recall)

    acc_arr = np.array(acc_arr)
    prec_arr = np.array(prec_arr)
    rec_arr = np.array(rec_arr)

    return np.mean(acc_arr), np.std(acc_arr), np.mean(prec_arr), np.std(prec_arr), np.mean(rec_arr), np.std(rec_arr)

def create_outputs(data_length):
    '''
    Creates the output vector.
    '''
    positive_out = np.empty(data_length)
    positive_out.fill(1)
    negative_out = np.empty(data_length)
    negative_out.fill(0)

    return positive_out, negative_out

def train_model(input_data, output_data, training_data_percentage, fetch_model, is_release=False):
    '''
    Executes supervised learning with the specified input and output data.
    :param input_data: The input data.
    :param output_data: The output data.
    :param training_data_percentage: The ratio of split between training and test set.
    :param fetch_model: The callback that will create a model.
    :param is_release: Whether the model is for the release or for experimenting.
    '''
    tokenizer = Tokenizer(nb_words=VOCAB_SIZE)

    tokenizer.fit_on_texts(input_data)

    if is_release is False:
        # Split the data set into training and test set.
        training_data_num_per_category = int(training_data_percentage * len(input_data) * 0.5)
        total_data_num_per_category = int(0.5 * len(input_data))

        training_data = input_data[:training_data_num_per_category] + input_data[total_data_num_per_category:(total_data_num_per_category + training_data_num_per_category)]
        test_data = input_data[training_data_num_per_category:total_data_num_per_category] + input_data[(total_data_num_per_category + training_data_num_per_category):]

        training_output = np.concatenate((output_data[:training_data_num_per_category], output_data[total_data_num_per_category:(total_data_num_per_category + training_data_num_per_category)]), axis=None)
        test_output = np.concatenate((output_data[training_data_num_per_category:total_data_num_per_category], output_data[(total_data_num_per_category + training_data_num_per_category):]), axis=None)
    else:
        # For the final model use all the data to train.
        training_data = input_data
        training_output = output_data

    # Convert it to be numpy array.
    training_output = np.array(training_output)

    if is_release is False:
        test_output = np.array(test_output)

    # Encode the words to be represented as unique numbers.
    encoded_training_input = tokenizer.texts_to_sequences(training_data)

    if is_release is False:
        encoded_test_input = tokenizer.texts_to_sequences(test_data)

    # Pad the sequence so it has the same length for each sample.
    padded_training_input = pad_sequences(encoded_training_input, maxlen=MAX_LENGTH, padding='post')
    
    if is_release is False:
        padded_test_input = pad_sequences(encoded_test_input, maxlen=MAX_LENGTH, padding='post')

    # Fetch the model we want to train.
    model = fetch_model()

    # Define callbacks for early stop.
    callbacks = [CustomEarlyStopping(val_acc_th=0.85, acc_th=0.85, min_epochs=20, verbose=1)] 

    history = model.fit(padded_training_input, training_output, epochs=70, validation_split=0.15, verbose=2, callbacks=callbacks)

    # Find validation accuracy values through the training, and use plot to display the graph.
    val_acc_arr = history.history['loss']
    acc_arr = history.history['val_loss']

    generate_embeddings(model, tokenizer)

    if is_release is False:
        loss, accuracy, precision, recall = model.evaluate(padded_test_input, test_output, verbose=2)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        return model, accuracy, precision, recall

    return model