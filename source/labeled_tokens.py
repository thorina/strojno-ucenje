import csv
import os

from nltk.corpus import names

LABELED_TOKENS = '../data/trained-models/labeled_tokens.txt'
LABELED_TOKENS_PUNCT = '../data/trained-models/labeled_tokens_punct.txt'
LABELED_TOKENS_LOWER = '../data/trained-models/labeled_tokens_lower.txt'
LABELED_TOKENS_LOWER_PUNCT = '../data/trained-models/labeled_tokens_lower_punct.txt'
TAGGED_FILES_PATH = '../data/training-data'


def populate_labeled_tokens(punctuation, lowercase):
    show_message(lowercase, punctuation)
    path = get_path(punctuation, lowercase)

    if lowercase:
        male_names = [(name.lower(), 'C') for name in names.words('male.txt')]
        female_names = [(name.lower(), 'C') for name in names.words('female.txt')]
    else:
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
                    label = line[0]
                    token = line[1]
                    if punctuation:
                        if lowercase:
                            story_tokens += [(token.lower(), label)]
                        else:
                            story_tokens += [(token, label)]
                    else:
                        if line[1].isalnum():
                            if lowercase:
                                story_tokens += [(token.lower(), label)]
                            else:
                                story_tokens += [(token, label)]

            labeled_tokens += story_tokens

    with open(path, 'w') as file_output:
        file_output.write(repr(labeled_tokens))

    print('Labeled tokens populated!')
    return labeled_tokens


def show_message(lowercase, punctuation):
    if punctuation:
        if lowercase:
            print("Populating labeled lowercase tokens with punctuation...")
        else:
            print("Populating labeled tokens with punctuation...")
    else:
        if lowercase:
            print("Populating labeled lowercase tokens without punctuation...")
        else:
            print("Populating labeled tokens without punctuation...")


def load_labeled_tokens(punctuation, lowercase):
    path = get_path(lowercase, punctuation)

    if os.path.exists(path):
        with open(path, 'r') as file_input:
            labeled_tokens = eval(file_input.read())
        print('Loaded existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        print('Generating new training data and new file.')
        labeled_tokens = populate_labeled_tokens(punctuation, lowercase)

    return labeled_tokens


def get_path(lowercase, punctuation):
    if punctuation:
        if lowercase:
            path = LABELED_TOKENS_LOWER_PUNCT
        else:
            path = LABELED_TOKENS_PUNCT
    else:
        if lowercase:
            path = LABELED_TOKENS_LOWER
        else:
            path = LABELED_TOKENS
    return path
