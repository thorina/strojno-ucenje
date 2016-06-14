import os

import nltk


CV_FILES_PATH = '../data/cross-validation'
CV_RESULTS_FILES_PATH = CV_FILES_PATH + '/' + 'current-results'

TRAINED_CRF_MODEL_CV = CV_RESULTS_FILES_PATH + '/' + 'trained_crf.model'
TRAINED_CRF_PUNCT_MODEL_CV = CV_RESULTS_FILES_PATH + '/' + 'trained_crf_punct.model'
TRAINED_CRF_LOWER_MODEL_CV = CV_RESULTS_FILES_PATH + '/' + 'trained_crf_lower.model'
TRAINED_CRF_LOWER_PUNCT_MODEL_CV = CV_RESULTS_FILES_PATH + '/' + 'trained_crf_lower_punct.model'

TRAINED_MODELS = '../data/trained-models'
TRAINED_CRF_MODEL = TRAINED_MODELS + '/' + 'trained_crf.model'
TRAINED_CRF_PUNCT_MODEL = TRAINED_MODELS + '/' + 'trained_crf_punct.model'
TRAINED_CRF_LOWER_MODEL = TRAINED_MODELS + '/' + 'trained_crf_lower.model'
TRAINED_CRF_LOWER_PUNCT_MODEL = TRAINED_MODELS + '/' + 'trained_crf_lower_punct.model'


def load_trained_crf_model(punctuation, lowercase):
    path = get_path(punctuation, lowercase, False)
    if os.path.exists(path):
        crf = nltk.CRFTagger()
        crf.set_model_file(path)
        print('Loaded existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        print('Training new model and creating new file.')
        return

    return crf


def train_crf_model(labeled_names, punctuation, lowercase):
    crf = nltk.CRFTagger()
    training_data = [labeled_names]
    path = get_path(punctuation, lowercase, True)
    crf.train(training_data, path)
    return crf


def get_path(lowercase, punctuation, cv):

    if cv:
        if punctuation:
            if lowercase:
                path = TRAINED_CRF_LOWER_PUNCT_MODEL_CV
            else:
                path = TRAINED_CRF_PUNCT_MODEL_CV
        else:
            if lowercase:
                path = TRAINED_CRF_LOWER_MODEL_CV
            else:
                path = TRAINED_CRF_MODEL_CV

    else:
        if punctuation:
            if lowercase:
                path = TRAINED_CRF_LOWER_PUNCT_MODEL
            else:
                path = TRAINED_CRF_PUNCT_MODEL
        else:
            if lowercase:
                path = TRAINED_CRF_LOWER_MODEL
            else:
                path = TRAINED_CRF_MODEL
    return path


