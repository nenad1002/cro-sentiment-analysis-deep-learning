import csv
import re
import numpy as np

from sentimental import test_models, create_outputs, train_model, predict
from models import define_and_compile_model2
from stemmer import process_row, has_vowel, transform, convert_to_root, remove_croatian_symbols, STOP_WORDS


def main():
    with open("input-small.csv", encoding="utf8") as csv_file:
        '''
        Opens the input csv file and divides it into positive and negative reviews.
        '''
        csv_reader = csv.reader(csv_file, delimiter=',')
        positives = []
        negatives = []
        for row in csv_reader:
            if row[1] == '1':
                positives.append(row[0])
            if row[1] == '0':
                negatives.append(row[0])

        stem_positive = []
        stem_negative = []

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in positives:
            stem_positive.append(process_row(row))

        for row in negatives:
            stem_negative.append(process_row(row))

        positive_out, negative_out = create_outputs(len(stem_positive))

        train_model(stem_positive + stem_negative, np.concatenate((positive_out, negative_out), axis=None), training_data_percentage=0.85, fetch_model=define_and_compile_model2, is_release=True)

        # Predict an example.
        print (predict("Uzivao sam prilikom ovog smjestaja i vratiti cu se.", "trained_model_data"))


if __name__ == "__main__":
    main()