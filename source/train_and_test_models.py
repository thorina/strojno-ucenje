import csv
import os
import re
from collections import Counter

from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models

TEST_FILES_PATH = '../data/test-files'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'


def tag_file_with_all_models(file_name, models):
    path = TEST_FILES_PATH + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File ' + path + ' does not exist!')
        return

    content = get_content(path)
    content_lowercase = content.lower()
    tokenized_content = wordpunct_tokenize(content)
    tokenized_content_punct = word_tokenize(content)
    tokenized_content_lowercase = wordpunct_tokenize(content_lowercase)
    tokenized_content_lowercase_punct = word_tokenize(content_lowercase)

    print('\nTagging content with HMM without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.hmm, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.hmm_punct, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lowercase, models.hmm_lowercase, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lowercase_punct, models.hmm_lowercase_punct, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.crf, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.crf_punct, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lowercase, models.crf_lowercase, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lowercase_punct, models.crf_lowercase_punct, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)


def tag_tokens_with_model(tokens, model, lowercase):
    tagged_content = model.tag(tokens)

    characters = []
    words = [word for word, tag in tagged_content]
    for word, tag in tagged_content:
        # we do not want to add both Queen and queen to characters, only queen
        # we want to add Hercules as Hercules
        if tag == 'C':
            if lowercase:
                characters.append(word)
            else:
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
        for token, label in tagged_content:
            csv_writer.writerow([label] + [token])
        print('File ' + tagged_file_path + ' created!')


def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
    return content


def main():
    print('Do you want to retrain the models? y/n')
    print('(do this if you are running this for the first time or if training set has changed)')
    print('If you want to exit, enter q.')
    while True:
        input_string = input('--> ')
        if input_string == 'q':
            quit()

        models = Models()
        if input_string == 'y':
            models.retrain_models()
            break

        elif input_string == 'n':
            models.load_all_trained_models()
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
            quit()
        tag_file_with_all_models(input_string, models)


if __name__ == "__main__":
    main()
