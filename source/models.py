import csv
import os
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.conditional_random_fields import train_crf_model, load_trained_crf_model
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model
from source.labeled_tokens import populate_labeled_tokens, load_labeled_tokens

TEST_FILES_PATH = '../data/test-files'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


# convention:
# suffix _punct for everything that includes punctuation
# no suffix for everything that ignores punctuation

def tag_file_with_all_models(file_name, hmm, hmm_punct, crf, crf_punct):
    path = TEST_FILES_PATH + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File does not exist!')
        return

    content = get_content(path)
    tokenized_content = wordpunct_tokenize(content)
    tokenized_content_punct = word_tokenize(content)

    print('\nTagging content with HMM without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, hmm)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, hmm_punct)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, crf)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, crf_punct)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)


def tag_tokens_with_model(tokens, model):
    tagged_content = model.tag(tokens)

    characters = []
    words = [word for word, tag in tagged_content]
    for word, tag in tagged_content:
        # we do not want to add both Queen and queen to characters, only queen
        # we want to add Hercules as Hercules
        if tag == 'C':
            if word.lower() not in words:
                characters.append(word)
            else:
                characters.append(word.lower())

    counter = Counter(characters)
    print('Found characters and number of their occurrences:')
    for character, occurrences in counter.items():
        print(character + ' ' + str(occurrences))

    return tagged_content


def write_tagged_content_to_file(tagged_content, tagged_file_path):
    with open(tagged_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        for word in tagged_content:
            csv_writer.writerow(['O'] + [word])
        print('File ' + tagged_file_path + ' created!')


def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
    return content


def load_all_trained_models():
    labeled_names = load_labeled_tokens(False)
    labeled_names_punct = load_labeled_tokens(True)
    hmm = load_trained_hmm_model(labeled_names, False)
    hmm_punct = load_trained_hmm_model(labeled_names_punct, True)
    crf = load_trained_crf_model(labeled_names, False)
    crf_punct = load_trained_crf_model(labeled_names_punct, True)
    return crf, crf_punct, hmm, hmm_punct


def retrain_models():
    labeled_tokens = populate_labeled_tokens(True)
    labeled_tokens_punct = populate_labeled_tokens(False)
    hmm = train_hmm_model(labeled_tokens, False)
    hmm_punct = train_hmm_model(labeled_tokens_punct, True)
    crf = train_crf_model(labeled_tokens, False)
    crf_punct = train_crf_model(labeled_tokens_punct, True)
    return crf, crf_punct, hmm, hmm_punct


def main():
    print('Do you want to retrain the models? y/n')
    print('(do this if you are running this for the first time or if training set has changed)')
    print('If you want to exit, enter q.')
    while True:
        input_string = input('--> ')
        if input_string == 'q':
            quit()

        if input_string == 'y':
            crf, crf_punct, hmm, hmm_punct = retrain_models()
            break

        elif input_string == 'n':
            crf, crf_punct, hmm, hmm_punct = load_all_trained_models()
            break

        else:
            print('Incorrect input - please enter y, n or q.')
            continue

    while True:
        print('\n')
        print('Enter file name without extension in /data/test-data to tag, e.g. "1" for file "1.txt".')
        print('If you want to exit, enter q.')
        input_string = input('--> ')
        if input_string == 'q':
            break
        tag_file_with_all_models(input_string, hmm, hmm_punct, crf, crf_punct)


if __name__ == "__main__":
    main()
