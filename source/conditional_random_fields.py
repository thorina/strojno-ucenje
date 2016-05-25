import os

import nltk

TRAINED_CRF_MODEL = '../data/trained_crf.model'
TRAINED_CRF_MODEL_PUNCT = '../data/trained_crf_punct.model'


def load_trained_crf_model(labeled_names, punctuation):
    if punctuation:
        path = TRAINED_CRF_MODEL_PUNCT
    else:
        path = TRAINED_CRF_MODEL

    if os.path.exists(path):
        crf = nltk.CRFTagger()
        crf.set_model_file(path)
        print('Loaded existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        print('Training new model and creating new file.')
        crf = train_crf_model(labeled_names, punctuation)
    return crf


def train_crf_model(labeled_names, punctuation):
    crf = nltk.CRFTagger()
    training_data = [labeled_names]
    if punctuation:
        print("Training CRF with punctuation...")
        path = TRAINED_CRF_MODEL_PUNCT
    else:
        print("Training CRF without punctuation...")
        path = TRAINED_CRF_MODEL

    crf.train(training_data, path)
    print("CRF trained!")
    return crf
