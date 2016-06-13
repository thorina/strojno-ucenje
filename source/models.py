import os

import nltk
from nltk.tag import StanfordNERTagger

from source.conditional_random_fields import train_crf_model, load_trained_crf_model
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model
from source.labeled_tokens import populate_labeled_tokens,part_populate_labeled_tokens
from source.utils import write_tagged_content_to_file

STANFORD_NER_JAR = '../lib/stanford-ner/stanford-ner.jar'
STANFORD_NER_MODELS_PATH = '../lib/stanford-ner/classifiers/'
TRAINED_STANFORD_NER_LOWER_PUNCT = 'trained_stanford_ner_lower_punct.ser.gz'
TRAINED_STANFORD_NER_LOWER = 'trained_stanford_ner_lower.ser.gz'
TRAINED_STANFORD_NER_PUNCT = 'trained_stanford_ner_punct.ser.gz'
TRAINED_STANFORD_NER = 'trained_stanford_ner.ser.gz'

TEST_FILES_PATH = '../data/test-files/stories'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'

PATH_TOKENIZED_CONTENT = '../lib/stanford-ner/training-sets/tokenized_content.tsv'
PATH_TOKENIZED_CONTENT_PUNCT = '../lib/stanford-ner/training-sets/tokenized_content_punct.tsv'
PATH_TOKENIZED_CONTENT_LOWER = '../lib/stanford-ner/training-sets/tokenized_content_lower.tsv'
PATH_TOKENIZED_CONTENT_LOWER_PUNCT = '../lib/stanford-ner/training-sets/tokenized_content_lower_punct.tsv'

NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)


class Models:
    hmm = None
    hmm_lower = None
    hmm_punct = None
    hmm_lower_punct = None
    crf = None
    crf_lower = None
    crf_punct = None
    crf_lower_punct = None
    stanford_ner = None
    stanford_ner_lower = None
    stanford_ner_punct = None
    stanford_ner_lower_punct = None

    def load_all_trained_models(self):
        labeled_tokens = populate_labeled_tokens(False, False)
        labeled_tokens_punct = populate_labeled_tokens(True, False)
        labeled_tokens_lower = populate_labeled_tokens(False, True)
        labeled_tokens_lower_punct = populate_labeled_tokens(True, True)

        self.hmm = load_trained_hmm_model(labeled_tokens, False, False)
        self.hmm_punct = load_trained_hmm_model(labeled_tokens_punct, True, False)
        self.hmm_lower = load_trained_hmm_model(labeled_tokens_lower, False, True)
        self.hmm_lower_punct = load_trained_hmm_model(labeled_tokens_lower_punct, True, True)

        self.crf = load_trained_crf_model(labeled_tokens, False, False)
        self.crf_punct = load_trained_crf_model(labeled_tokens_punct, True, False)
        self.crf_lower = load_trained_crf_model(labeled_tokens_lower, False, True)
        self.crf_lower_punct = load_trained_crf_model(labeled_tokens_lower_punct, True, True)

        self.load_existing_stanford_ner_models()

    def retrain_models(self):
        labeled_tokens = populate_labeled_tokens(False, False)
        labeled_tokens_punct = populate_labeled_tokens(True, False)
        labeled_tokens_lower = populate_labeled_tokens(False, True)
        labeled_tokens_lower_punct = populate_labeled_tokens(True, True)

        self.hmm = train_hmm_model(labeled_tokens, False, False)
        self.hmm_punct = train_hmm_model(labeled_tokens_punct, True, False)
        self.hmm_lower = train_hmm_model(labeled_tokens_lower, False, True)
        self.hmm_lower_punct = train_hmm_model(labeled_tokens_lower_punct, True, True)

        self.crf = train_crf_model(labeled_tokens, False, False)
        self.crf_punct = train_crf_model(labeled_tokens_punct, True, False)
        self.crf_lower = train_crf_model(labeled_tokens_lower, False, True)
        self.crf_lower_punct = train_crf_model(labeled_tokens_lower_punct, True, True)

        # Stanford NER needs all training data as single tsv file
        write_tagged_content_to_file(labeled_tokens, PATH_TOKENIZED_CONTENT)
        write_tagged_content_to_file(labeled_tokens_punct, PATH_TOKENIZED_CONTENT_PUNCT)
        write_tagged_content_to_file(labeled_tokens_lower, PATH_TOKENIZED_CONTENT_LOWER)
        write_tagged_content_to_file(labeled_tokens_lower_punct, PATH_TOKENIZED_CONTENT_LOWER_PUNCT)

        show_retraining_instruction()
        show_bash_instruction(TRAINED_STANFORD_NER)
        show_bash_instruction(TRAINED_STANFORD_NER_PUNCT)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER_PUNCT)
        self.wait_until_models_are_trained()

    def retrain_hmm_models(self, list_dir):
        labeled_tokens = part_populate_labeled_tokens(list_dir, False, False)
        labeled_tokens_punct = part_populate_labeled_tokens(list_dir, True, False)
        labeled_tokens_lower = part_populate_labeled_tokens(list_dir, False, True)
        labeled_tokens_lower_punct = part_populate_labeled_tokens(list_dir, True, True)

        self.hmm = train_hmm_model(labeled_tokens, False, False)
        self.hmm_punct = train_hmm_model(labeled_tokens_punct, True, False)
        self.hmm_lower = train_hmm_model(labeled_tokens_lower, False, True)
        self.hmm_lower_punct = train_hmm_model(labeled_tokens_lower_punct, True, True)

    def retrain_crf_models(self, list_dir):
        labeled_tokens = part_populate_labeled_tokens(list_dir, False, False)
        labeled_tokens_punct = part_populate_labeled_tokens(list_dir, True, False)
        labeled_tokens_lower = part_populate_labeled_tokens(list_dir, False, True)
        labeled_tokens_lower_punct = part_populate_labeled_tokens(list_dir, True, True)

        self.crf = train_crf_model(labeled_tokens, False, False)
        self.crf_punct = train_crf_model(labeled_tokens_punct, True, False)
        self.crf_lower = train_crf_model(labeled_tokens_lower, False, True)
        self.crf_lower_punct = train_crf_model(labeled_tokens_lower_punct, True, True)

    def retrain_ner_models(self, list_dir):
        labeled_tokens = part_populate_labeled_tokens(list_dir, False, False)
        labeled_tokens_punct = part_populate_labeled_tokens(list_dir, True, False)
        labeled_tokens_lower = part_populate_labeled_tokens(list_dir, False, True)
        labeled_tokens_lower_punct = part_populate_labeled_tokens(list_dir, True, True)

        # Stanford NER needs all training data as single tsv file
        write_tagged_content_to_file(labeled_tokens, PATH_TOKENIZED_CONTENT)
        write_tagged_content_to_file(labeled_tokens_punct, PATH_TOKENIZED_CONTENT_PUNCT)
        write_tagged_content_to_file(labeled_tokens_lower, PATH_TOKENIZED_CONTENT_LOWER)
        write_tagged_content_to_file(labeled_tokens_lower_punct, PATH_TOKENIZED_CONTENT_LOWER_PUNCT)

        show_retraining_instruction()
        show_bash_instruction(TRAINED_STANFORD_NER)
        show_bash_instruction(TRAINED_STANFORD_NER_PUNCT)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER_PUNCT)

    def load_existing_stanford_ner_models(self):
        needs_training = check_if_all_models_exist()
        if needs_training:
            self.wait_until_models_are_trained()
        else:
            self.load_stanford_ner_models()

    def load_stanford_ner_models(self):
        self.stanford_ner = StanfordNERTagger(
            STANFORD_NER_MODELS_PATH + TRAINED_STANFORD_NER,
            STANFORD_NER_JAR,
            encoding='utf-8')

        self.stanford_ner_punct = StanfordNERTagger(
            STANFORD_NER_MODELS_PATH + TRAINED_STANFORD_NER_PUNCT,
            STANFORD_NER_JAR,
            encoding='utf-8')

        self.stanford_ner_lower = StanfordNERTagger(
            STANFORD_NER_MODELS_PATH + TRAINED_STANFORD_NER_LOWER,
            STANFORD_NER_JAR,
            encoding='utf-8')

        self.stanford_ner_lower_punct = StanfordNERTagger(
            STANFORD_NER_MODELS_PATH + TRAINED_STANFORD_NER_LOWER_PUNCT,
            STANFORD_NER_JAR,
            encoding='utf-8')

    def wait_until_models_are_trained(self):
        print('Please enter ok when training finishes.')
        print('If you want to exit, enter q.')
        while True:
            input_string = input('--> ')
            if input_string == 'q':
                quit()
            elif input_string == 'ok':
                self.load_existing_stanford_ner_models()
                break
            else:
                print('Incorrect input - please enter ok or q.')
                continue


def model_does_not_exist(model):
    path = STANFORD_NER_MODELS_PATH + model
    if not os.path.exists(path):
        show_warning(model)
        return True


def check_if_all_models_exist():
    ner_missing = model_does_not_exist(TRAINED_STANFORD_NER)
    ner_punct_missing = model_does_not_exist(TRAINED_STANFORD_NER_PUNCT)
    ner_lower_missing = model_does_not_exist(TRAINED_STANFORD_NER_LOWER)
    ner_lower_punct_missing = model_does_not_exist(TRAINED_STANFORD_NER_LOWER_PUNCT)
    return ner_missing or ner_punct_missing or ner_lower_missing or ner_lower_punct_missing


def show_warning(model):
    print('File ' + model + ' does not exist!')
    show_retraining_instruction()
    show_bash_instruction(model)


def show_retraining_instruction():
    print('You should now retrain Stanford NER tagger with following command(s) from terminal '
          'run from directory ../lib/stanford-ner:')


def show_bash_instruction(model):
    print('java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier '
          '-prop ner.properties -trainFile training-sets/tokenized_content.tsv '
          '-serializeTo ' + model + '\n')
