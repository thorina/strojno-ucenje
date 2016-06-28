import os

import shutil

import numpy as np
from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models, calculate_confidence_matrix_for_file
from source.utils import write_tagged_content_to_file, get_content, tag_tokens_with_model, get_all_tags, parse_tsv

TRAINED_MODELS = '../data/trained-models'
TEST_FILES_PATH = '../data/test/stories'
TEST_RESULTS = '../data/test/results'
TAGGED_TEST_FILES_PATH = '../data/test/tagged'
OUR_TAGGED_TEST_FILES_PATH = '../data/test/our_tag'
TRAINING_DATA = '../data/training-data'
ORIGINAL_STORIES = '../data/stories'


def tag_file_with_all_models(file_name, models):
    path = ORIGINAL_STORIES + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File ' + path + ' does not exist!')
        return

    content = get_content(path)
    content_lower = content.lower()
    tokenized_content = wordpunct_tokenize(content)
    tokenized_content_punct = word_tokenize(content)
    tokenized_content_lower = wordpunct_tokenize(content_lower)
    tokenized_content_lower_punct = word_tokenize(content_lower)

    print('\nTagging content with HMM without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.hmm,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with HMM with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.hmm_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with HMM with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.hmm_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with HMM with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.hmm_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.crf,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.crf_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.crf_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.crf_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.stanford_ner,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.stanford_ner_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.stanford_ner_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.stanford_ner_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)


def test(models):
    print('Models will now be tested...')
    if os.path.exists(TEST_RESULTS):
        shutil.rmtree(TEST_RESULTS)
    os.makedirs(TEST_RESULTS)

    file_list = os.listdir(TEST_FILES_PATH)
    for file in file_list:
        tag_file_with_all_models(file.replace(".txt", ""), models)

    conf_matrix_hmm = np.array([[0, 0], [0, 0]])
    conf_matrix_hmm_punct = np.array([[0, 0], [0, 0]])
    conf_matrix_hmm_lower = np.array([[0, 0], [0, 0]])
    conf_matrix_hmm_lower_punct = np.array([[0, 0], [0, 0]])
    conf_matrix_crf = np.array([[0, 0], [0, 0]])
    conf_matrix_crf_punct = np.array([[0, 0], [0, 0]])
    conf_matrix_crf_lower = np.array([[0, 0], [0, 0]])
    conf_matrix_crf_lower_punct = np.array([[0, 0], [0, 0]])
    conf_matrix_stanford_ner = np.array([[0, 0], [0, 0]])
    conf_matrix_stanford_ner_punct = np.array([[0, 0], [0, 0]])
    conf_matrix_stanford_ner_lower = np.array([[0, 0], [0, 0]])
    conf_matrix_stanford_ner_lower_punct = np.array([[0, 0], [0, 0]])

    tagged_file_list = os.listdir(TAGGED_TEST_FILES_PATH)
    for file in tagged_file_list:
        tagged_model = file[file.find("_") + 1: file.find(".")]
        tagged_name = file[: file.find("_")]
        machine_tags = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + file)
        our_tags = parse_tsv(OUR_TAGGED_TEST_FILES_PATH + '/' + tagged_name + '.tsv')

        new_matrix = calculate_confidence_matrix_for_file(machine_tags, our_tags,
                                                          tagged_name, TEST_RESULTS + '/' + tagged_model + '.txt')

        if tagged_model == 'hmm':
            conf_matrix_hmm += new_matrix
        elif tagged_model == 'hmm_punct':
            conf_matrix_hmm_punct += new_matrix
        elif tagged_model == 'hmm_lower':
            conf_matrix_hmm_lower += new_matrix
        elif tagged_model == 'hmm_lower_punct':
            conf_matrix_hmm_lower_punct += new_matrix
        elif tagged_model == 'crf':
            conf_matrix_crf += new_matrix
        elif tagged_model == 'crf_punct':
            conf_matrix_crf_punct += new_matrix
        elif tagged_model == 'crf_lower':
            conf_matrix_crf_lower += new_matrix
        elif tagged_model == 'crf_lower_punct':
            conf_matrix_crf_lower_punct += new_matrix
        elif tagged_model == 'stanford_ner':
            conf_matrix_stanford_ner += new_matrix
        elif tagged_model == 'stanford_ner_punct':
            conf_matrix_stanford_ner_punct += new_matrix
        elif tagged_model == 'stanford_ner_lower':
            conf_matrix_stanford_ner_lower += new_matrix
        elif tagged_model == 'stanford_ner_lower_punct':
            conf_matrix_stanford_ner_lower_punct += new_matrix

    write_confidence_matrix_and_f_score(conf_matrix_hmm, 'hmm')
    write_confidence_matrix_and_f_score(conf_matrix_hmm_punct, 'hmm_punct')
    write_confidence_matrix_and_f_score(conf_matrix_hmm_lower, 'hmm_lower')
    write_confidence_matrix_and_f_score(conf_matrix_hmm_lower_punct, 'hmm_lower_punct')

    write_confidence_matrix_and_f_score(conf_matrix_crf, 'crf')
    write_confidence_matrix_and_f_score(conf_matrix_crf_punct, 'crf_punct')
    write_confidence_matrix_and_f_score(conf_matrix_crf_lower, 'crf_lower')
    write_confidence_matrix_and_f_score(conf_matrix_crf_lower_punct, 'crf_lower_punct')

    write_confidence_matrix_and_f_score(conf_matrix_stanford_ner, 'stanford_ner')
    write_confidence_matrix_and_f_score(conf_matrix_stanford_ner_punct, 'stanford_ner_punct')
    write_confidence_matrix_and_f_score(conf_matrix_stanford_ner_lower, 'stanford_ner_lower')
    write_confidence_matrix_and_f_score(conf_matrix_stanford_ner_lower_punct, 'stanford_ner_lower_punct')


def write_confidence_matrix_and_f_score(confidence_matrix, model):
    with open(TEST_RESULTS + '/' + model + '.txt', 'a') as f:
        f.write('Final results:\n')
        f.write(np.array_str(confidence_matrix))
        f.write('\n')
        f.write('F_2 score:\n')

        beta = 2  # važnija nam je osjetljivost od preciznosti
        tp = confidence_matrix[0][0]
        fn = confidence_matrix[0][1]
        fp = confidence_matrix[1][0]
        fscore = ((1 + beta * beta) * tp) / ((1 + beta * beta) * tp + beta * beta * fn + fp)
        print()
        print('Model: ' + model)
        print('Confidence matrix:')
        print(confidence_matrix)
        print('F_2 score = ' + str(fscore))
        print()
        f.write(str(fscore))


def main():
    print('Do you want to retrain the models? "y"/"n"')
    print('(do this if you are running this for the first time or if training set has changed)')
    print('If you want to exit, enter "q".')

    if not os.path.exists(TRAINED_MODELS):
        os.makedirs(TRAINED_MODELS)

    while True:
        input_string = input('--> ')
        if input_string.lower() == 'q':
            quit()

        models = Models()
        if input_string.lower() == 'y':
            models.retrain_all_models()
            test(models)
            break

        elif input_string.lower() == 'n':
            models.load_all_trained_models()
            # test(models)
            break

        else:
            print('Incorrect input - please enter "y", "n" or "q".')
            continue

    if not os.path.exists(TAGGED_TEST_FILES_PATH):
        os.makedirs(TAGGED_TEST_FILES_PATH)

    while True:
        print()
        print('Enter file name without extension in /data/stories to tag, e.g. "1" for file "1.txt".')
        print('If you want to exit, enter q.')
        input_string = input('--> ')
        if input_string.lower() == 'q':
            quit()
        else:
            tag_file_with_all_models(input_string, models)


if __name__ == "__main__":
    main()
