import csv
import os
import re

import dill
import nltk
from nltk.corpus import names
from nltk.tokenize import word_tokenize, wordpunct_tokenize

TEST_DATA_PATH = '../data/test-data'

TRAINED_HMM_MODEL_PUNCT_PICKLE = '../data/trained_hmm_with_punctuation.dill'
TRAINED_HMM_MODEL_PICKLE = '../data/trained_hmm_no_punctuation.dill'
TRAINED_CRF_MODEL_PICKLE = '../data/trained_crf_no_punctuation.dill'
TRAINED_CRF_MODEL_PUNCT_PICKLE = '../data/trained_crf_with_punctuation.dill'
LABELED_NAMES_WITH_PUNCTUATION = '../data/labeled_data_punctuation.txt'
LABELED_NAMES_WITHOUT_PUNCT = '../data/labeled_data_no_punctuation.txt'

NER_TRAINING_DATA_PATH = '../data/training-data'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


def populate_labeled_names_without_punctuation():
    print("Populating labeled names without punctuation...")

    male_names = [(name, 'C') for name in names.words('male.txt')]
    female_names = [(name, 'C') for name in names.words('female.txt')]
    labeled_names = male_names + female_names

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
                    if line[1].isalnum():
                        story += [(x, y)]
            labeled_names += story

    return labeled_names


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


def train_hmm_model_without_punctuation(labeled_names):

    # # uncomment if you want to get labeled data again and store it in files
    # labeled_names = populate_labeled_names_without_punctuation()
    # with open(LABELED_NAMES_WITHOUT_PUNCT, 'w') as file_output:
    #     file_output.write(repr(labeled_names))

    # uncomment if you want to read already populated data from files
    # with open(LABELED_NAMES_WITHOUT_PUNCT, 'r') as file_input:
    #     labeled_names = eval(file_input.read())

    # uncomment if you want to train new hmm model
    print("Training HMM that does not include punctuation...")
    # states = ["O", "C"]
    # symbols = list(set([ss[0] for sss in labeled_names for ss in sss]))
    # hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    # hmm_no_punct = hmm_trainer.train_supervised([labeled_names])
    # with open(TRAINED_HMM_MODEL_PICKLE, 'wb') as file_output:
    #     dill.dump(hmm_no_punct, file_output)

    #uncomment if you want to read trained models from files
    with open(TRAINED_HMM_MODEL_PICKLE, 'rb') as file_output:
        hmm_no_punct = dill.load(file_output)
    print("HMM trained!")
    return hmm_no_punct


def train_hmm_model_with_punctuation(labeled_names_with_punctuation):
    # # uncomment if you want to get labeled data again and store it in files
    # labeled_names_with_punctuation = populate_labeled_names_with_punctuation()
    # with open(LABELED_NAMES_WITH_PUNCTUATION, 'w') as file_output:
    #     file_output.write(repr(labeled_names_with_punctuation))

    # uncomment if you want to read already populated data from files
    # with open(LABELED_NAMES_WITH_PUNCTUATION, 'r') as file_input:
    #     labeled_names_with_punctuation = eval(file_input.read())

    # uncomment if you want to train new hmm model
    print("Training HMM that includes punctuation...")
    # states = ["O", "C"]
    # symbols = list(set([ss[0] for sss in labeled_names_with_punctuation for ss in sss]))
    # hmm_trainer_punct = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    # hmm_with_punct = hmm_trainer_punct.train_supervised([labeled_names_with_punctuation])
    # with open(TRAINED_HMM_MODEL_PUNCT_PICKLE, 'wb') as file_output:
    #     dill.dump(hmm_with_punct, file_output)

    # uncomment if you want to load model from file
    with open(TRAINED_HMM_MODEL_PUNCT_PICKLE, 'rb') as file_output:
        hmm_with_punct = dill.load(file_output)
    print("HMM trained!")
    return hmm_with_punct


def train_crf(labeled_names, punctuation):
    crf = nltk.CRFTagger()
    training_data = [labeled_names]
    if punctuation:
        print("Training CRF with punctuation...")
        path = TRAINED_CRF_MODEL_PICKLE
    else:
        print("Training CRF without punctuation...")
        path = TRAINED_CRF_MODEL_PUNCT_PICKLE

    crf.train(training_data, 'model.crf.tagger')
    with open(path, 'wb') as file_output:
        dill.dump(crf, file_output)
    print("CRF trained!")
    return crf


def test_models(file_name, hmm_with_punct, hmm_without_punct, crf_with_punct, crf_without_punct):
    file_name += '.txt'
    if file_name not in os.listdir(TEST_DATA_PATH):
        print('File does not exist!')
        return

    with open(TEST_DATA_PATH + '/' + file_name, 'r') as file_output:
        content = file_output.read()
        content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
        tokenized_content_with_punct = word_tokenize(content)
        tokenized_content_without_punct = wordpunct_tokenize(content)
        print("Tagging content with HMM with punctuation...")
        tagged_content = hmm_with_punct.tag(tokenized_content_with_punct)
        print('Found characters:')
        characters = set()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with HMM without punctuation...")
        tagged_content = hmm_without_punct.tag(tokenized_content_without_punct)
        print('Found characters:')
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with CRF with punctuation...")
        tokenized_content_with_punct = word_tokenize(content)
        tagged_content = crf_with_punct.tag(tokenized_content_with_punct)
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with CRF without punctuation...")
        tokenized_content_with_punct = wordpunct_tokenize(content)
        tagged_content = crf_without_punct.tag(tokenized_content_without_punct)
        print('Found characters:')
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

if __name__ == "__main__":

    # labeled_names_with_punct = populate_labeled_names_with_punctuation()
    # labeled_names_without_punct = populate_labeled_names_without_punctuation()

    # # uncomment if you want to get labeled data again and store it in files
    # labeled_names = populate_labeled_names_without_punctuation()
    # with open(LABELED_NAMES_WITHOUT_PUNCT, 'w') as file_output:
    #     file_output.write(repr(labeled_names))

    # uncomment if you want to read already populated data from files
    with open(LABELED_NAMES_WITHOUT_PUNCT, 'r') as file_input:
        labeled_names_without_punct = eval(file_input.read())
    with open(LABELED_NAMES_WITH_PUNCTUATION, 'r') as file_input:
        labeled_names_with_punct = eval(file_input.read())

    hmm_with_punct = train_hmm_model_with_punctuation(labeled_names_with_punct)
    hmm_without_punct = train_hmm_model_without_punctuation(labeled_names_without_punct)
    crf_with_punct = train_crf(labeled_names_with_punct, True)
    crf_without_punct = train_crf(labeled_names_without_punct, False)
    while True:
        print('\n')
        print('Enter file name in /data/test-data to tag, e.g. "1" for file "1.txt".')
        print('If you want to exit, enter q.')
        input_string = input('--> ')
        if input_string == 'q':
            break
        test_models(input_string, hmm_with_punct, hmm_without_punct, crf_with_punct, crf_without_punct)
