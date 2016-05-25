import os
import re

import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.conditional_random_fields import train_crf_model, load_trained_crf_model
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model
from source.labeled_tokens import populate_labeled_tokens, load_labeled_tokens

TEST_DATA_PATH = '../data/test-data'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


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
        characters = set()
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
        test_models(input_string, hmm, hmm_punct, crf, crf_punct)


if __name__ == "__main__":
    main()
