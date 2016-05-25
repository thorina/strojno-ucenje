import os
import dill
from nltk.tag.hmm import HiddenMarkovModelTrainer


TRAINED_HMM_MODEL = '../data/trained_hmm.dill'
TRAINED_HMM_MODEL_PUNCT = '../data/trained_hmm_punct.dill'


def load_trained_hmm_model(labeled_names, punctuation):
    if punctuation:
        path = TRAINED_HMM_MODEL_PUNCT
    else:
        path = TRAINED_HMM_MODEL

    if os.path.exists(path):
        with open(path, 'rb') as file_output:
            hmm = dill.load(file_output)
        print('Loaded existing file ' + path)
    else:
        print('File ' + path + ' does not exist!')
        print('Training new model and creating new file.')
        hmm = train_hmm_model(labeled_names, punctuation)

    return hmm


def train_hmm_model(labeled_names, punctuation):
    if punctuation:
        print("Training HMM with punctuation...")
        path = TRAINED_HMM_MODEL_PUNCT

    else:
        print("Training HMM without punctuation...")
        path = TRAINED_HMM_MODEL

    states = ["O", "C"]
    symbols = list(set([ss[0] for sss in labeled_names for ss in sss]))
    hmm_trainer = HiddenMarkovModelTrainer(states=states, symbols=symbols)
    hmm = hmm_trainer.train_supervised([labeled_names])

    with open(path, 'wb') as file_output:
        dill.dump(hmm, file_output)

    print("HMM trained!")
    return hmm

