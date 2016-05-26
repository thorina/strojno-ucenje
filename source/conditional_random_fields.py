import os

import nltk

TRAINED_CRF_MODEL = '../data/trained-models/trained_crf.model'
TRAINED_CRF_PUNCT_MODEL = '../data/trained-models/trained_crf_punct.model'
TRAINED_CRF_LOWERCASE_MODEL = '../data/trained-models/trained_crf_lowercase.model'
TRAINED_CRF_LOWERCASE_PUNCT_MODEL = '../data/trained-models/trained_crf_lowercase_punct.model'


def load_trained_crf_model(labeled_names, punctuation, lowercase):

    path = get_path(punctuation, lowercase)
    if os.path.exists(path):
        crf = nltk.CRFTagger()
        crf.set_model_file(path)
        print('Loaded existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        crf = train_crf_model(labeled_names, punctuation, lowercase)
    return crf


def train_crf_model(labeled_names, punctuation, lowercase):
    crf = nltk.CRFTagger()
    training_data = [labeled_names]
    show_message(punctuation, lowercase)
    path = get_path(punctuation, lowercase)
    crf.train(training_data, path)
    print("CRF trained!")
    return crf


def get_path(lowercase, punctuation):
    if punctuation:
        if lowercase:
            path = TRAINED_CRF_LOWERCASE_PUNCT_MODEL
        else:
            path = TRAINED_CRF_PUNCT_MODEL
    else:
        if lowercase:
            path = TRAINED_CRF_LOWERCASE_MODEL
        else:
            path = TRAINED_CRF_MODEL
    return path


def show_message(lowercase, punctuation):
    if punctuation:
        if lowercase:
            print("Training CRF with lowercase tokens and with punctuation...")
        else:
            print("Training CRF with original tokens and with punctuation...")
    else:
        if lowercase:
            print("Training CRF with lowercase tokens and without punctuation...")
        else:
            print("Training CRF with original tokens and without punctuation...")
