import csv
import os
import re

import dill
import nltk
from nltk.corpus import names
from nltk.tokenize import word_tokenize, wordpunct_tokenize

TEST_DATA_PATH = '../data/test-data'

TRAINED_HMM_MODEL = '../data/trained_hmm.dill'
TRAINED_HMM_MODEL_PUNCT = '../data/trained_hmm_punct.dill'
TRAINED_CRF_MODEL = '../data/trained_crf.model'
TRAINED_CRF_MODEL_PUNCT = '../data/trained_crf_punct.model'
LABELED_NAMES = '../data/labeled_names.txt'
LABELED_NAMES_PUNCT = '../data/labeled_names_punct.txt'

TAGGED_FILES_PATH = '../data/correctly-tagged-tsv-files'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


def populate_labeled_names(punctuation):
    if punctuation:
        print("Populating labeled names with punctuation...")
        path = LABELED_NAMES
    else:
        print("Populating labeled names without punctuation...")
        path = LABELED_NAMES_PUNCT

    male_names = [(name, 'C') for name in names.words('male.txt')]
    female_names = [(name, 'C') for name in names.words('female.txt')]
    labeled_names = male_names + female_names

    for filename in os.listdir(TAGGED_FILES_PATH):
        story = []
        with open(TAGGED_FILES_PATH + '/' + filename) as tsv:
            reader = csv.reader(tsv, dialect="excel-tab")
            for line in reader:
                if len(line) < 2:
                    print("tsv error in " + filename + " at line " + str(reader.line_num) + ': ' + str(line))
                else:
                    y = line[0]
                    x = line[1]
                    if punctuation:
                        story += [(x, y)]
                    else:
                        if line[1].isalnum():
                            story += [(x, y)]
            labeled_names += story

    with open(path, 'w') as file_output:
        file_output.write(repr(labeled_names))
    return labeled_names


def train_hmm_model(labeled_names, punctuation):
    # uncomment if you want to train new hmm model
    if punctuation:
        print("Training HMM with punctuation...")
        path = TRAINED_HMM_MODEL_PUNCT

    else:
        print("Training HMM without punctuation...")
        path = TRAINED_HMM_MODEL

    states = ["O", "C"]
    symbols = list(set([ss[0] for sss in labeled_names for ss in sss]))
    hmm_trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
    hmm = hmm_trainer.train_supervised([labeled_names])

    with open(path, 'wb') as file_output:
        dill.dump(hmm, file_output)

    print("HMM trained!")
    return hmm


def train_crf_model(labeled_names, punctuation):
    crf = nltk.CRFTagger()
    training_data = [labeled_names]
    if punctuation:
        print("Training CRF with punctuation...")
        path = TRAINED_CRF_MODEL
    else:
        print("Training CRF without punctuation...")
        path = TRAINED_CRF_MODEL_PUNCT

    crf.train(training_data, path)
    print("CRF trained!")
    return crf


def test_models(file_name, hmm, hmm_punct, crf, crf_punct):
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
        tagged_content = hmm_punct.tag(tokenized_content_with_punct)
        print('Found characters:')
        characters = set()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with HMM without punctuation...")
        tagged_content = hmm.tag(tokenized_content_without_punct)
        print('Found characters:')
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with CRF with punctuation...")
        tokenized_content_with_punct = word_tokenize(content)
        tagged_content = crf_punct.tag(tokenized_content_with_punct)
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)

        print("\nTagging content with CRF without punctuation...")
        tokenized_content_without_punct = wordpunct_tokenize(content)
        tagged_content = crf.tag(tokenized_content_without_punct)
        print('Found characters:')
        characters.clear()
        for word, tag in tagged_content:
            if tag == 'C':
                characters.add(word)
        print(characters)


def main():

    print('Do you want to generate new training data from tagged tsv files? y/n')
    print('(do this if those files have changed or new files have been added)')
    print('If you want to exit, enter q.')
    while True:
        input_string = input('--> ')
        if input_string == 'q':
            quit()

        if input_string == 'y':
            labeled_names = populate_labeled_names(True)
            labeled_names_punct = populate_labeled_names(False)
            break

        elif input_string == 'n':
            if os.path.exists(LABELED_NAMES):
                with open(LABELED_NAMES, 'r') as file_input:
                    labeled_names = eval(file_input.read())
                print('Read existing file /data/labeled_data.txt.')
            else:
                print('File /data/labeled_data.txt does not exist!')
                print('Generating new training data and new file.')
                labeled_names = populate_labeled_names(False)

            if os.path.exists(LABELED_NAMES_PUNCT):
                with open(LABELED_NAMES_PUNCT, 'r') as file_input:
                    labeled_names_punct = eval(file_input.read())
                print('Read existing file /data/labeled_data_punct.txt.')
            else:
                print('File /data/labeled_data_punct.txt does not exist!')
                print('Generating new training data and new file.')
                labeled_names_punct = populate_labeled_names(True)
            break
        else:
            print('Incorrect input - please enter y, n or q.')
            continue

    print('\n')
    print('Do you want to train new HMM taggers? y/n')
    print('(do this if the training data has changed)')
    print('If you want to exit, enter q.')
    while True:
        input_string = input('--> ')
        if input_string == 'q':
            quit()

        if input_string == 'y':
            hmm = train_hmm_model(labeled_names, False)
            hmm_punct = train_hmm_model(labeled_names_punct, True)
            break

        elif input_string == 'n':
            if os.path.exists(TRAINED_HMM_MODEL):
                with open(TRAINED_HMM_MODEL, 'rb') as file_output:
                    hmm = dill.load(file_output)
                print('Loaded existing file /data/trained_hmm.dill.')

            else:
                print('File /data/trained_hmm.dill does not exist!')
                print('Training new model and creating new file.')
                hmm = train_hmm_model(labeled_names, False)

            if os.path.exists(TRAINED_HMM_MODEL_PUNCT):
                with open(TRAINED_HMM_MODEL_PUNCT, 'rb') as file_output:
                    hmm_punct = dill.load(file_output)
                print('Loaded existing file /data/trained_hmm_punct.dill.')

            else:
                print('File /data/trained_hmm_punct.dill does not exist!')
                print('Training new model and creating new file.')
                hmm_punct = train_hmm_model(labeled_names_punct, True)
            break

        else:
            print('Incorrect input - please enter y, n or q.')
            continue

    print('\n')
    print('Do you want to train new CRF taggers? y/n')
    print('(do this if the training data has changed)')
    print('If you want to exit, enter q.')
    while True:
        input_string = input('--> ')
        if input_string == 'q':
            quit()

        if input_string == 'y':
            crf = train_crf_model(labeled_names, False)
            crf_punct = train_crf_model(labeled_names_punct, True)
            break

        elif input_string == 'n':
            if os.path.exists(TRAINED_CRF_MODEL):
                crf = nltk.CRFTagger()
                crf.set_model_file(TRAINED_CRF_MODEL)
                print('Loaded existing file /data/trained_crf.model.')
            else:
                print('File /data/trained_crf.model does not exist!')
                print('Training new model and creating new file.')
                crf = train_crf_model(labeled_names, False)

            if os.path.exists(TRAINED_CRF_MODEL_PUNCT):
                crf_punct = nltk.CRFTagger()
                crf.set_model_file(TRAINED_CRF_MODEL_PUNCT)
                print('Loaded existing file /data/trained_crf_punct.model.')
            else:
                print('File /data/trained_crf_punct.model does not exist!')
                print('Training new model and creating new file.')
                crf_punct = train_crf_model(labeled_names_punct, True)
            break

        else:
            print('Incorrect input - please enter y, n or q.')
            continue

    while True:
        print('\n')
        print('Enter file name in /data/test-data to tag, e.g. "1" for file "1.txt".')
        print('If you want to exit, enter q.')
        input_string = input('--> ')
        if input_string == 'q':
            break
        test_models(input_string, hmm, hmm_punct, crf, crf_punct)


if __name__ == "__main__":
    main()
