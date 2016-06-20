import os
import shutil

import nltk
import numpy as np
from dill import dill
from nltk.tag import StanfordNERTagger
import matplotlib.pyplot as plt

from source.conditional_random_fields import load_trained_crf_model, train_crf_model, TRAINED_CRF_MODEL_CV, \
    TRAINED_CRF_PUNCT_MODEL_CV, TRAINED_CRF_LOWER_MODEL_CV, TRAINED_CRF_LOWER_PUNCT_MODEL_CV, TRAINED_CRF_MODEL, \
    TRAINED_CRF_PUNCT_MODEL, TRAINED_CRF_LOWER_MODEL, TRAINED_CRF_LOWER_PUNCT_MODEL
from source.hidden_markov_model import train_hmm_model, load_trained_hmm_model, TRAINED_HMM_MODEL, \
    TRAINED_HMM_PUNCT_MODEL, TRAINED_HMM_LOWER_PUNCT_MODEL, TRAINED_HMM_LOWER_MODEL
from source.labeled_tokens import populate_labeled_tokens
from source.utils import write_tagged_content_to_file, tag_tokens_with_model, get_content, get_all_tags, parse_tsv, \
    TemporaryModels, FScores, ConfidenceMatrices

CV_FILES_PATH = '../data/cross-validation'
CV_FILES_PATH_LOWER_PUNCT = CV_FILES_PATH + '/' + 'lower_punct'
CV_FILES_PATH_LOWER = CV_FILES_PATH + '/' + 'lower'
CV_FILES_PATH_PUNCT = CV_FILES_PATH + '/' + 'punct'
CV_FILES_PATH_DEFAULT = CV_FILES_PATH + '/' + 'default'
CV_RESULTS_FILES_PATH = CV_FILES_PATH + '/' + 'current-results'
CV_HMM_RESULTS = CV_FILES_PATH + '/' + 'hmm'
CV_CRF_RESULTS = CV_FILES_PATH + '/' + 'crf'

TRAINING_DATA = '../data/training-data'
TEST_FILES_PATH = '../data/test/stories'
ORIGINAL_STORIES = '../data/stories'

STANFORD_NER_JAR = '../lib/stanford-ner/stanford-ner.jar'
STANFORD_NER_MODELS_PATH = '../lib/stanford-ner/classifiers/'
TRAINED_STANFORD_NER_LOWER_PUNCT = 'trained_stanford_ner_lower_punct.ser.gz'
TRAINED_STANFORD_NER_LOWER = 'trained_stanford_ner_lower.ser.gz'
TRAINED_STANFORD_NER_PUNCT = 'trained_stanford_ner_punct.ser.gz'
TRAINED_STANFORD_NER = 'trained_stanford_ner.ser.gz'

PATH_TOKENIZED_CONTENT = '../lib/stanford-ner/training-sets/tokenized_content.tsv'
PATH_TOKENIZED_CONTENT_PUNCT = '../lib/stanford-ner/training-sets/tokenized_content_punct.tsv'
PATH_TOKENIZED_CONTENT_LOWER = '../lib/stanford-ner/training-sets/tokenized_content_lower.tsv'
PATH_TOKENIZED_CONTENT_LOWER_PUNCT = '../lib/stanford-ner/training-sets/tokenized_content_lower_punct.tsv'

NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def plot_cv_data(iter, array, model):
    y_def = []
    y_lower = []
    y_punct = []
    y_lower_punct = []
    x = [i for i in range(1, iter + 1)]
    for i in range(len(array)):
        y_def += [array[i].default]
        y_punct += [array[i].punct]
        y_lower += [array[i].lower]
        y_lower_punct += [array[i].lower_punct]

    print(type(y_def), y_def)
    fig, ax = plt.subplots()
    ax.plot(x, y_def, label=r"default")
    ax.plot(x, y_lower, label=r"lower")
    ax.plot(x, y_punct, label=r"punct")
    ax.plot(x, y_lower_punct, label=r"lower_punct")
    ax.set_xlabel('iteracija')
    ax.set_ylabel('F score')
    ax.legend(loc=2)
    ax.set_title(model)

    fig.savefig(CV_FILES_PATH + '/' + model + '.png')


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
        self.hmm = load_trained_hmm_model(False, False)
        self.hmm_punct = load_trained_hmm_model(True, False)
        self.hmm_lower = load_trained_hmm_model(False, True)
        self.hmm_lower_punct = load_trained_hmm_model(True, True)
        self.crf = load_trained_crf_model(False, False)
        self.crf_punct = load_trained_crf_model(True, False)
        self.crf_lower = load_trained_crf_model(False, True)
        self.crf_lower_punct = load_trained_crf_model(True, True)
        self.load_existing_stanford_ner_models()

    def retrain_all_models(self):

        if not os.path.exists(CV_HMM_RESULTS):
            os.makedirs(CV_HMM_RESULTS)
        if not os.path.exists(CV_CRF_RESULTS):
            os.makedirs(CV_CRF_RESULTS)
        if not os.path.exists(CV_RESULTS_FILES_PATH):
            os.makedirs(CV_RESULTS_FILES_PATH)

        all_files = os.listdir(TRAINING_DATA)
        cv_size = 3
        iterations = len(os.listdir(TRAINING_DATA)) // cv_size
        print('Number of folds: ' + str(iterations))
        fscores_hmm = FScores()
        fscores_crf = FScores()

        array_crf = []
        array_hmm = []

        for i in range(0, iterations):
            if os.path.exists(CV_RESULTS_FILES_PATH):
                shutil.rmtree(CV_RESULTS_FILES_PATH)
            os.makedirs(CV_RESULTS_FILES_PATH)

            test_files, tmp_models_hmm, tmp_models_crf = train_models(all_files, cv_size, i)
            array_hmm += [self.test_models(fscores_hmm, i, 'hmm', test_files, tmp_models_hmm)]
            array_crf += [self.test_models(fscores_crf, i, 'crf', test_files, tmp_models_crf)]

        plot_cv_data(iterations, array_crf, 'crf')
        plot_cv_data(iterations, array_hmm, 'hmm')

        # # remove unnecessary folders after CV
        shutil.rmtree(CV_FILES_PATH_DEFAULT)
        shutil.rmtree(CV_FILES_PATH_PUNCT)
        shutil.rmtree(CV_FILES_PATH_LOWER)
        shutil.rmtree(CV_FILES_PATH_LOWER_PUNCT)
        shutil.rmtree(CV_RESULTS_FILES_PATH)

        print('Stanford NER needs to be run manually from terminal, so it does not implement cross-validation.')
        all_training_files = os.listdir(TRAINING_DATA)
        labeled_tokens = populate_labeled_tokens(all_training_files, False, False)
        labeled_tokens_punct = populate_labeled_tokens(all_training_files, True, False)
        labeled_tokens_lower = populate_labeled_tokens(all_training_files, False, True)
        labeled_tokens_lower_punct = populate_labeled_tokens(all_training_files, True, True)

        # Stanford NER needs all training data as single tsv file
        write_tagged_content_to_file(labeled_tokens, PATH_TOKENIZED_CONTENT, message=True)
        write_tagged_content_to_file(labeled_tokens_punct, PATH_TOKENIZED_CONTENT_PUNCT, message=True)
        write_tagged_content_to_file(labeled_tokens_lower, PATH_TOKENIZED_CONTENT_LOWER, message=True)
        write_tagged_content_to_file(labeled_tokens_lower_punct, PATH_TOKENIZED_CONTENT_LOWER_PUNCT, message=True)

        show_retraining_instruction()
        show_bash_instruction(TRAINED_STANFORD_NER)
        show_bash_instruction(TRAINED_STANFORD_NER_PUNCT)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER)
        show_bash_instruction(TRAINED_STANFORD_NER_LOWER_PUNCT)
        self.wait_until_models_are_trained()

    def test_models(self, fscores_max, i, model, test_files, tmp_models):
        for j in range(0, len(test_files)):
            test_files[j] = test_files[j].replace('.tsv', '')

        print('Tagging cross-validation files with trained models...')
        tag_files_for_cross_validation(test_files, tmp_models)
        confidence_matrices = calculate_cv_confidence_matrices(model)
        fscores_tmp = calculate_f_scores(confidence_matrices)

        if fscores_tmp.default > fscores_max.default:
            fscores_max.default = fscores_tmp.default
            print('Found new best ' + model.upper() + ' without punctuation in iteration ' + str(i) +
                  ', with F_2 score of ' + str(fscores_max.default))

            if model == 'hmm':
                self.hmm = tmp_models.default
                results_path = CV_HMM_RESULTS
                path = TRAINED_HMM_MODEL
                with open(path, 'wb') as file_output:
                    dill.dump(self.hmm, file_output)

            else:
                self.crf = tmp_models.default
                results_path = CV_CRF_RESULTS
                tmp_model_path = TRAINED_CRF_MODEL_CV
                path = TRAINED_CRF_MODEL
                shutil.move(tmp_model_path, path)

            shutil.move(CV_RESULTS_FILES_PATH + '/' + model + '.txt',
                        results_path + '/' + model + '.txt')

        if fscores_tmp.punct > fscores_max.punct:
            fscores_max.punct = fscores_tmp.punct
            print('Found new best ' + model.upper() + ' with punctuation in iteration ' + str(i) +
                  ', with F_2 score of ' + str(fscores_max.default))

            if model == 'hmm':
                self.hmm_punct = tmp_models.punct
                results_path = CV_HMM_RESULTS
                path = TRAINED_HMM_PUNCT_MODEL
                with open(path, 'wb') as file_output:
                    dill.dump(self.hmm_punct, file_output)
            else:
                self.crf_punct = tmp_models.punct
                results_path = CV_CRF_RESULTS
                tmp_model_path = TRAINED_CRF_PUNCT_MODEL_CV
                path = TRAINED_CRF_PUNCT_MODEL
                shutil.move(tmp_model_path, path)

            shutil.move(CV_RESULTS_FILES_PATH + '/' + model + '_punct.txt',
                        results_path + '/' + model + '_punct.txt')

        if fscores_tmp.lower > fscores_max.lower:
            fscores_max.lower = fscores_tmp.lower
            print('Found new best ' + model.upper() + ' without punctuation with lowercase tokens in iteration ' + str(
                i) +
                  ', with F_2 score of ' + str(fscores_max.default))
            if model == 'hmm':
                self.hmm_lower = tmp_models.lower
                results_path = CV_HMM_RESULTS
                path = TRAINED_HMM_LOWER_MODEL
                with open(path, 'wb') as file_output:
                    dill.dump(self.hmm_lower, file_output)
            else:
                self.crf_lower = tmp_models.lower
                results_path = CV_CRF_RESULTS
                tmp_model_path = TRAINED_CRF_LOWER_MODEL_CV
                path = TRAINED_CRF_LOWER_MODEL
                shutil.move(tmp_model_path, path)

            shutil.move(CV_RESULTS_FILES_PATH + '/' + model + '_lower.txt',
                        results_path + '/' + model + '_lower.txt')

        if fscores_tmp.lower_punct > fscores_max.lower_punct:
            fscores_max.lower_punct = fscores_tmp.lower_punct
            print('Found new best ' + model.upper() + ' with punctuation and with lowercase tokens in iteration ' + str(
                i) +
                  ', with F_2 score of ' + str(fscores_max.lower_punct))

            if model == 'hmm':
                self.hmm_lower_punct = tmp_models.lower_punct
                results_path = CV_HMM_RESULTS
                path = TRAINED_HMM_LOWER_PUNCT_MODEL
                with open(path, 'wb') as file_output:
                    dill.dump(self.hmm_lower_punct, file_output)
            else:
                self.crf_lower_punct = tmp_models.lower_punct
                results_path = CV_CRF_RESULTS
                tmp_model_path = TRAINED_CRF_LOWER_PUNCT_MODEL_CV
                path = TRAINED_CRF_LOWER_PUNCT_MODEL
                shutil.move(tmp_model_path, path)

            shutil.move(CV_RESULTS_FILES_PATH + '/' + model + '_lower_punct.txt',
                        results_path + '/' + model + '_lower_punct.txt')
        return fscores_tmp

    def load_existing_stanford_ner_models(self):
        needs_training = check_if_all_stanford_ner_models_exist()
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

        print('Stanford NER models loaded!')

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


def train_models(all_files, cv_size, i):
    print()
    print('Iteration ' + str(i + 1) + ' running...')
    train_files = all_files[:i] + all_files[i + cv_size:]
    test_files = all_files[i:i + cv_size]
    print('Getting tokens...')
    labeled_tokens = populate_labeled_tokens(train_files, False, False)
    labeled_tokens_punct = populate_labeled_tokens(train_files, True, False)
    labeled_tokens_lower = populate_labeled_tokens(train_files, False, True)
    labeled_tokens_lower_punct = populate_labeled_tokens(train_files, True, True)

    print('Training HMM models...')
    tmp_models_hmm = TemporaryModels()
    tmp_models_hmm.default = train_hmm_model(labeled_tokens)
    tmp_models_hmm.punct = train_hmm_model(labeled_tokens_punct)
    tmp_models_hmm.lower = train_hmm_model(labeled_tokens_lower)
    tmp_models_hmm.lower_punct = train_hmm_model(labeled_tokens_lower_punct)

    print('Training CRF models...')
    tmp_models_crf = TemporaryModels()
    tmp_models_crf.default = train_crf_model(labeled_tokens, False, False)
    tmp_models_crf.punct = train_crf_model(labeled_tokens_punct, True, False)
    tmp_models_crf.lower = train_crf_model(labeled_tokens_lower, False, True)
    tmp_models_crf.lower_punct = train_crf_model(labeled_tokens_lower_punct, True, True)

    return test_files, tmp_models_hmm, tmp_models_crf


def stanford_ner_model_does_not_exist(model):
    path = STANFORD_NER_MODELS_PATH + model
    if not os.path.exists(path):
        show_warning(model)
        return True


def check_if_all_stanford_ner_models_exist():
    ner_missing = stanford_ner_model_does_not_exist(TRAINED_STANFORD_NER)
    ner_punct_missing = stanford_ner_model_does_not_exist(TRAINED_STANFORD_NER_PUNCT)
    ner_lower_missing = stanford_ner_model_does_not_exist(TRAINED_STANFORD_NER_LOWER)
    ner_lower_punct_missing = stanford_ner_model_does_not_exist(TRAINED_STANFORD_NER_LOWER_PUNCT)
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
          '-serializeTo classifiers/' + model)


def tag_files_for_cross_validation(file_list, tmp_models):
    # first clean CV files folder
    if os.path.exists(CV_FILES_PATH_DEFAULT):
        shutil.rmtree(CV_FILES_PATH_DEFAULT)
    if os.path.exists(CV_FILES_PATH_PUNCT):
        shutil.rmtree(CV_FILES_PATH_PUNCT)
    if os.path.exists(CV_FILES_PATH_LOWER):
        shutil.rmtree(CV_FILES_PATH_LOWER)
    if os.path.exists(CV_FILES_PATH_LOWER_PUNCT):
        shutil.rmtree(CV_FILES_PATH_LOWER_PUNCT)

    # then create new CV folders
    os.makedirs(CV_FILES_PATH_DEFAULT)
    os.makedirs(CV_FILES_PATH_PUNCT)
    os.makedirs(CV_FILES_PATH_LOWER)
    os.makedirs(CV_FILES_PATH_LOWER_PUNCT)

    for file_name in file_list:
        path = ORIGINAL_STORIES + '/' + file_name + '.txt'

        if not os.path.isfile(path):
            print('File ' + path + ' does not exist!')
            continue

        content = get_content(path)
        content_lower = content.lower()
        tokenized_content = nltk.wordpunct_tokenize(content)
        tokenized_content_punct = nltk.word_tokenize(content)
        tokenized_content_lower = nltk.wordpunct_tokenize(content_lower)
        tokenized_content_lower_punct = nltk.word_tokenize(content_lower)

        tagged_content = tag_tokens_with_model(tokenized_content, tmp_models.default, lowercase=False, message=False)
        tagged_file_path = CV_FILES_PATH_DEFAULT + '/' + file_name + '.tsv'
        write_tagged_content_to_file(tagged_content, tagged_file_path, message=False)

        tagged_content = tag_tokens_with_model(tokenized_content_punct, tmp_models.punct, lowercase=False,
                                               message=False)
        tagged_file_path = CV_FILES_PATH_PUNCT + '/' + file_name + '.tsv'
        write_tagged_content_to_file(tagged_content, tagged_file_path, message=False)

        tagged_content = tag_tokens_with_model(tokenized_content_lower, tmp_models.lower, lowercase=True, message=False)
        tagged_file_path = CV_FILES_PATH_LOWER + '/' + file_name + '.tsv'
        write_tagged_content_to_file(tagged_content, tagged_file_path, message=False)

        tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, tmp_models.lower_punct, lowercase=True,
                                               message=False)
        tagged_file_path = CV_FILES_PATH_LOWER_PUNCT + '/' + file_name + '.tsv'
        write_tagged_content_to_file(tagged_content, tagged_file_path, message=False)


def calculate_confidence_matrix_for_file(machine_tag, our_tag, story, path):
    machine_tag_c_set = get_all_tags(machine_tag, 'C')
    our_tag_c_set = get_all_tags(our_tag, 'C')
    our_tag_c_set_copy = our_tag_c_set[:]
    our_tag_o_set = get_all_tags(our_tag, 'O')

    conf_matrix = np.array([[0, 0], [0, 0]])

    for word in machine_tag_c_set:
        if word in our_tag_c_set_copy:
            conf_matrix[0][0] += 1
            our_tag_c_set_copy.remove(word)
        else:
            conf_matrix[1][0] += 1

    conf_matrix[0][1] = len(our_tag_c_set_copy)
    # other su oznaceni kao other - oni koji nisu trebali biti other
    conf_matrix[1][1] = len(our_tag_o_set) - conf_matrix[1][0]
    write_to_file(path, story, machine_tag_c_set, our_tag_c_set, conf_matrix)
    return conf_matrix


def write_to_file(path, story, machine_tag, our_tag, conf_mtr):
    f = open(path, 'a')
    f.write(story + ':\n')
    f.write('comp tag: ' + ' '.join(machine_tag))
    f.write('\n')
    f.write('our tag: ' + ' '.join(our_tag))
    f.write('\n')
    f.write(np.array_str(conf_mtr))
    f.write('\n\n\n')
    f.close()


def calculate_cv_confidence_matrices(model):
    confidence_mtr = ConfidenceMatrices()

    path_default = CV_RESULTS_FILES_PATH + '/' + model + '.txt'
    path_punct = CV_RESULTS_FILES_PATH + '/' + model + '_punct.txt'
    path_lower = CV_RESULTS_FILES_PATH + '/' + model + '_lower.txt'
    path_lower_punct = CV_RESULTS_FILES_PATH + '/' + model + '_lower_punct.txt'

    for file in os.listdir(CV_FILES_PATH + '/' + 'default'):
        path = CV_FILES_PATH_DEFAULT + '/' + file
        tag_default = parse_tsv(path)
        our_tag = parse_tsv(TRAINING_DATA + '/' + file)
        confidence_mtr.default += calculate_confidence_matrix_for_file(tag_default, our_tag, file, path_default)

    for file in os.listdir(CV_FILES_PATH + '/' + 'punct'):
        path = CV_FILES_PATH_PUNCT + '/' + file
        tag_punct = parse_tsv(path)
        our_tag = parse_tsv(TRAINING_DATA + '/' + file)
        confidence_mtr.punct += calculate_confidence_matrix_for_file(tag_punct, our_tag, file, path_punct)

    for file in os.listdir(CV_FILES_PATH + '/' + 'lower'):
        path = CV_FILES_PATH_LOWER + '/' + file
        tag_lower = parse_tsv(path)
        our_tag = parse_tsv(TRAINING_DATA + '/' + file)
        confidence_mtr.lower += calculate_confidence_matrix_for_file(tag_lower, our_tag, file, path_lower)

    for file in os.listdir(CV_FILES_PATH + '/' + 'lower_punct'):
        path = CV_FILES_PATH_LOWER_PUNCT + '/' + file
        tag_lower_punct = parse_tsv(path)
        our_tag = parse_tsv(TRAINING_DATA + '/' + file)
        confidence_mtr.lower_punct += calculate_confidence_matrix_for_file(tag_lower_punct, our_tag, file,
                                                                           path_lower_punct)

    write_to_file(path_default, 'CV', [], [], confidence_mtr.default)
    write_to_file(path_punct, 'CV', [], [], confidence_mtr.punct)
    write_to_file(path_lower, 'CV', [], [], confidence_mtr.lower)
    write_to_file(path_lower_punct, 'CV', [], [], confidence_mtr.lower_punct)

    return confidence_mtr


def calculate_f_scores(confidence_matrices):
    fscores = FScores()

    beta = 2  # važnija nam je osjetljivost od preciznosti
    tp = confidence_matrices.default[0][0]
    fn = confidence_matrices.default[0][1]
    fp = confidence_matrices.default[1][0]
    fscores.default = ((1 + beta * beta) * tp) / ((1 + beta * beta) * tp + beta * beta * fn + fp)

    tp = confidence_matrices.punct[0][0]
    fn = confidence_matrices.punct[0][1]
    fp = confidence_matrices.punct[1][0]
    fscores.punct = ((1 + beta * beta) * tp) / ((1 + beta * beta) * tp + beta * beta * fn + fp)

    tp = confidence_matrices.lower[0][0]
    fn = confidence_matrices.lower[0][1]
    fp = confidence_matrices.lower[1][0]
    fscores.lower = ((1 + beta * beta) * tp) / ((1 + beta * beta) * tp + beta * beta * fn + fp)

    tp = confidence_matrices.lower_punct[0][0]
    fn = confidence_matrices.lower_punct[0][1]
    fp = confidence_matrices.lower_punct[1][0]
    fscores.lower_punct = ((1 + beta * beta) * tp) / ((1 + beta * beta) * tp + beta * beta * fn + fp)

    return fscores


def retrain_ner_models(list_dir):
    labeled_tokens = populate_labeled_tokens(list_dir, False, False)
    labeled_tokens_punct = populate_labeled_tokens(list_dir, True, False)
    labeled_tokens_lower = populate_labeled_tokens(list_dir, False, True)
    labeled_tokens_lower_punct = populate_labeled_tokens(list_dir, True, True)

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
