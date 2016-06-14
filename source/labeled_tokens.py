import csv

from nltk.corpus import names

TAGGED_FILES_PATH = '../data/training-data'


def populate_labeled_tokens(list_dir, punctuation, lowercase):

    if lowercase:
        male_names = [(name.lower(), 'C') for name in names.words('male.txt')]
        female_names = [(name.lower(), 'C') for name in names.words('female.txt')]
    else:
        male_names = [(name, 'C') for name in names.words('male.txt')]
        female_names = [(name, 'C') for name in names.words('female.txt')]

    labeled_tokens = male_names + female_names

    for filename in list_dir:
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
