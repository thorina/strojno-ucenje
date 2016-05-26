import nltk

from source.conditional_random_fields import train_crf_model, load_trained_crf_model
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model
from source.labeled_tokens import populate_labeled_tokens

TEST_FILES_PATH = '../data/test-files'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


class Models:
    hmm = None
    hmm_lowercase = None
    hmm_punct = None
    hmm_lowercase_punct = None
    crf = None
    crf_lowercase = None
    crf_punct = None
    crf_lowercase_punct = None

    def load_all_trained_models(self):
        labeled_tokens = populate_labeled_tokens(False, False)
        labeled_tokens_punct = populate_labeled_tokens(True, False)
        labeled_tokens_lowercase = populate_labeled_tokens(False, True)
        labeled_tokens_lowercase_punct = populate_labeled_tokens(True, False)

        self.hmm = load_trained_hmm_model(labeled_tokens, False, False)
        self.hmm_punct = load_trained_hmm_model(labeled_tokens_punct, True, False)
        self.hmm_lowercase = load_trained_hmm_model(labeled_tokens_lowercase, False, True)
        self.hmm_lowercase_punct = load_trained_hmm_model(labeled_tokens_lowercase_punct, True, True)

        self.crf = load_trained_crf_model(labeled_tokens, False, False)
        self.crf_punct = load_trained_crf_model(labeled_tokens_punct, True, False)
        self.crf_lowercase = load_trained_crf_model(labeled_tokens_lowercase, False, True)
        self.crf_lowercase_punct = load_trained_crf_model(labeled_tokens_lowercase_punct, True, True)

    def retrain_models(self):
        labeled_tokens = populate_labeled_tokens(False, False)
        labeled_tokens_punct = populate_labeled_tokens(True, False)
        labeled_tokens_lowercase = populate_labeled_tokens(False, True)
        labeled_tokens_lowercase_punct = populate_labeled_tokens(True, False)

        self.hmm = train_hmm_model(labeled_tokens, False, False)
        self.hmm_punct = train_hmm_model(labeled_tokens_punct, True, False)
        self.hmm_lowercase = train_hmm_model(labeled_tokens_lowercase, False, True)
        self.hmm_lowercase_punct = train_hmm_model(labeled_tokens_lowercase_punct, True, True)

        self.crf = train_crf_model(labeled_tokens, False, False)
        self.crf_punct = train_crf_model(labeled_tokens_punct, True, False)
        self.crf_lowercase = train_crf_model(labeled_tokens_lowercase, False, True)
        self.crf_lowercase_punct = train_crf_model(labeled_tokens_lowercase_punct, True, True)



