import nltk
from nltk.tag import StanfordNERTagger

from source.conditional_random_fields import train_crf_model, load_trained_crf_model
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model
from source.labeled_tokens import populate_labeled_tokens
from source.utils import write_tagged_content_to_file

TEST_FILES_PATH = '../data/test-files'
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

        self.load_stanford_ner_models()

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

        print('You should now retrain Stanford NER tagger with following commands from terminal '
              'run from directory ../lib/stanford-ner:')
        print('java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier '
              '-prop ner.properties -trainFile training-sets/tokenized_content.tsv '
              '-serializeTo classifiers/trained_stanford_ner.ser.gz')

        print('java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier '
              '-prop ner.properties -trainFile training-sets/tokenized_content_lower.tsv '
              '-serializeTo classifiers/trained_stanford_ner_lower.ser.gz')

        print('java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier '
              '-prop ner.properties -trainFile training-sets/tokenized_content_punct.tsv '
              '-serializeTo classifiers/trained_stanford_ner_punct.ser.gz')

        print('java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier '
              '-prop ner.properties -trainFile training-sets/tokenized_content_lower_punct.tsv '
              '-serializeTo classifiers/trained_stanford_ner_lower_punct.ser.gz')

        print('Please enter ok after the models are trained. If you enter ok before that, old models will be used!')
        print('If you want to exit, enter q.')
        while True:
            input_string = input('--> ')
            if input_string == 'q':
                quit()

            elif input_string == 'ok':
                self.load_stanford_ner_models()
                break

            else:
                print('Incorrect input - please enter ok or q.')
                continue

    def load_stanford_ner_models(self):
        self.stanford_ner = StanfordNERTagger('../lib/stanford-ner/classifiers/trained_stanford_ner.ser.gz',
                                              '../lib/stanford-ner/stanford-ner.jar',
                                              encoding='utf-8')

        self.stanford_ner_punct = StanfordNERTagger('../lib/stanford-ner/classifiers/trained_stanford_ner_lower.ser.gz',
                                                    '../lib/stanford-ner/stanford-ner.jar',
                                                    encoding='utf-8')

        self.stanford_ner_lower = StanfordNERTagger('../lib/stanford-ner/classifiers/trained_stanford_ner_lower.ser.gz',
                                                    '../lib/stanford-ner/stanford-ner.jar',
                                                    encoding='utf-8')

        self.stanford_ner_lower_punct = StanfordNERTagger(
            '../lib/stanford-ner/classifiers/trained_stanford_ner_lower_punct.ser.gz',
            '../lib/stanford-ner/stanford-ner.jar',
            encoding='utf-8')
