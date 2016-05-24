import csv
import os

import nltk
from nltk import CRFTagger
from nltk.corpus import names

NER_TRAINING_DATA_PATH = '../data/training-data'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


def populate_labeled_names_with_punctuation():
    print("Populating labeled names with punctuation...")
    male_names = [(name, 'C') for name in names.words('male.txt')]
    female_names = [(name, 'C') for name in names.words('female.txt')]
    labeled_names_with_punctuation = male_names + female_names

    for filename in os.listdir(NER_TRAINING_DATA_PATH):
        story = []
        with open(NER_TRAINING_DATA_PATH + '/' + filename) as tsv:
            reader = csv.reader(tsv, dialect="excel-tab")
            for line in reader:
                if len(line) < 2:
                    print("tsv error in " + filename + " at line " + str(reader.line_num) + ': ' + str(line))
                else:
                    y = line[0]
                    x = line[1]
                    story += [(x, y)]
            labeled_names_with_punctuation += story

    return labeled_names_with_punctuation


ct = CRFTagger()
training_data = []
training_data.append(populate_labeled_names_with_punctuation())
ct.train(training_data, 'model.crf.tagger')
while True:
    print('\n')
    print('If you want to exit, enter q.')
    input_string = input('--> ')
    if input_string == 'q':
        break
    print(ct.tag(["king"]))

