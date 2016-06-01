import os

import dill
from nltk.tag.hmm import HiddenMarkovModelTrainer

TRAINED_HMM_MODEL = '../data/trained-models/trained_hmm.dill'
TRAINED_HMM_PUNCT_MODEL = '../data/trained-models/trained_hmm_punct.dill'
TRAINED_HMM_LOWER_MODEL = '../data/trained-models/trained_hmm_lower.dill'
TRAINED_HMM_LOWER_PUNCT_MODEL = '../data/trained-models/trained_hmm_lower_punct.dill'


def load_trained_hmm_model(labeled_names, punctuation, lowercase):
    path = get_path(lowercase, punctuation)

    if os.path.exists(path):
        with open(path, 'rb') as file_output:
            hmm = dill.load(file_output)
        print('Loaded existing file ' + path)
    else:
        print('\nFile ' + path + ' does not exist!')
        print('Training new model and creating new file.')
        hmm = train_hmm_model(labeled_names, punctuation, lowercase)

    return hmm


def get_path(lowercase, punctuation):
    if punctuation:
        if lowercase:
            path = TRAINED_HMM_LOWER_PUNCT_MODEL
        else:
            path = TRAINED_HMM_PUNCT_MODEL
    else:
        if lowercase:
            path = TRAINED_HMM_LOWER_MODEL
        else:
            path = TRAINED_HMM_MODEL
    return path


def show_message(lowercase, punctuation):
    if punctuation:
        if lowercase:
            print("Training HMM with lowercase tokens and with punctuation...")
        else:
            print("Training HMM with original tokens and with punctuation...")
    else:
        if lowercase:
            print("Training HMM with lowercase tokens and without punctuation...")
        else:
            print("Training HMM with original tokens and without punctuation...")


def train_hmm_model(labeled_names, punctuation, lowercase):
    show_message(punctuation, lowercase)
    states = ["O", "C"]
    symbols = list(set([ss[0] for sss in labeled_names for ss in sss]))
    hmm_trainer = HiddenMarkovModelTrainer(states=states, symbols=symbols)
    hmm = hmm_trainer.train_supervised([labeled_names])

    path = get_path(punctuation, lowercase)
    with open(path, 'wb') as file_output:
        dill.dump(hmm, file_output)

    print("HMM trained!")
    return hmm
