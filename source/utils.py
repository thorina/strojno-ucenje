import csv
import re
from collections import Counter

import numpy as np


class FScores:
    default = -1
    punct = -1
    lower = -1
    lower_punct = -1


class ConfidenceMatrices:
    default = np.array([[0, 0], [0, 0]])
    punct = np.array([[0, 0], [0, 0]])
    lower = np.array([[0, 0], [0, 0]])
    lower_punct = np.array([[0, 0], [0, 0]])


class TemporaryModels:
    default = None
    punct = None
    lower = None
    lower_punct = None


def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = add_spaces_around_interpunctions(content)
    return content


def write_tagged_content_to_file(tagged_content, tagged_file_path, message):
    with open(tagged_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        for token, label in tagged_content:
            csv_writer.writerow([label] + [token])
        if message:
            print('File ' + tagged_file_path + ' created!')


def add_spaces_around_interpunctions(content):
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
    return content


def tag_tokens_with_model(tokens, model, lowercase, message):
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

    if message:
        counter = Counter(characters)
        print('Found characters and number of their occurrences: ')
        if len(characters) == 0:
            print('none')
        for character, occurrences in sorted(counter.items()):
            print(character + ' ' + str(occurrences))

    return tagged_content


def parse_tsv(file_tsv):
    return_list = []
    with open(file_tsv) as reader:
        reader = csv.reader(reader, dialect="excel-tab")
        for row in reader:
            if len(row) < 2:
                print("tsv error in " + file_tsv + " at line " + str(reader.line_num) + ': ' + str(row))
            else:
                return_list = return_list + [(row[0], row[1])]
    return return_list


def compare(tuple1, tuple2):
    if tuple1[0] == tuple2[0]:
        return 0
    return 1


def get_all_tags(tagged_list, letter):
    return_list = []
    for (tag, word) in tagged_list:
        word_lower = word.lower()
        # word_lower = word_lower.replace(';', '')
        if tag == letter and word_lower not in return_list and word_lower.isalnum() and \
                (len(word_lower) > 1 or word_lower == 'i'):
            return_list = return_list + [word_lower]
    return return_list
