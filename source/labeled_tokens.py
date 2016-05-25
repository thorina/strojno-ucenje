import csv
import os

from nltk.corpus import names

LABELED_TOKENS = '../data/labeled_tokens.txt'
LABELED_TOKENS_PUNCT = '../data/labeled_tokens_punct.txt'
TAGGED_FILES_PATH = '../data/correctly-tagged-tsv-files'


def populate_labeled_tokens(punctuation):
    if punctuation:
        print("Populating labeled tokens with punctuation...")
        path = LABELED_TOKENS
    else:
        print("Populating labeled tokens without punctuation...")
        path = LABELED_TOKENS_PUNCT

    male_names = [(name, 'C') for name in names.words('male.txt')]
    female_names = [(name, 'C') for name in names.words('female.txt')]
    labeled_tokens = male_names + female_names

    for filename in os.listdir(TAGGED_FILES_PATH):
        story_tokens = []
        with open(TAGGED_FILES_PATH + '/' + filename) as tsv:
            reader = csv.reader(tsv, dialect="excel-tab")
            for line in reader:
                if len(line) < 2:
                    print("tsv error in " + filename + " at line " + str(reader.line_num) + ': ' + str(line))
                else:
                    y = line[0]
                    x = line[1]
                    if punctuation:
                        story_tokens += [(x, y)]
                    else:
                        if line[1].isalnum():
                            story_tokens += [(x, y)]
            labeled_tokens += story_tokens

    with open(path, 'w') as file_output:
        file_output.write(repr(labeled_tokens))
    return labeled_tokens


def load_labeled_tokens(punctuation):
    if punctuation:
        path = LABELED_TOKENS_PUNCT
    else:
        path = LABELED_TOKENS

    if os.path.exists(path):
        with open(path, 'r') as file_input:
            labeled_tokens = eval(file_input.read())
        print('Read existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        print('Generating new training data and new file.')
        labeled_tokens = populate_labeled_tokens(punctuation)

    return labeled_tokens
