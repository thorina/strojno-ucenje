import csv
import os

import numpy as np
from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models
from source.utils import write_tagged_content_to_file, add_spaces_around_interpunctions

TRAINING_DATA = '../data/training-data'
TEST_FILES_PATH = '../data/test-files/stories'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'
ORIGINAL_STORIES = '../data/stories/'


# neka od tvojih funkcija iz train_and_test_models.py
def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = add_spaces_around_interpunctions(content)
    return content


# neka od tvojih funkcija iz train_and_test_models.py
def tag_tokens_with_model(tokens, model, lowercase):
    tagged_content = model.tag(tokens)

    characters = []
    words = [word for word, tag in tagged_content]
    for word, tag in tagged_content:
        # we do not want to add both Queen and queen to characters, only queen
        # we want to add Hercules as Hercules
        if tag == 'C':
            if lowercase:
                characters.append(word)
            else:
                if word.lower() not in words:
                    characters.append(word)
                else:
                    characters.append(word.lower())

    # counter = Counter(characters)
    # print('Found characters and number of their occurrences:')
    # for character, occurrences in counter.items():
        # print(character + ' ' + str(occurrences))

    return tagged_content


# preradjena funkcija tag_file_with_all_models
# koja radi sam na crf jer sam htio smanjiti vrijeme izvrsavanja
def tag_file_with_crf_model(file_name, model):
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

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, model.crf, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, model.crf_punct, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, model.crf_lower, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with CRF with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, model.crf_lower_punct, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)


def tag_file_with_hmm_model(file_name, model):
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
    tagged_content = tag_tokens_with_model(tokenized_content, model.hmm, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, model.hmm_punct, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, model.hmm_lower, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)

    print('\nTagging content with HMM with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, model.hmm_lower_punct, True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)


# funkcija za pretvaranje iz tsv u listu tuplova
def parse_tsv(file_tsv):
    return_list = []
    with open(file_tsv) as reader:
        read = csv.reader(reader, dialect="excel-tab")
        for row in read:
            return_list = return_list + [(row[0], row[1])]
    return return_list


# usporedba dali su im rijeci jednake
def compare(touple1, touple2):
    if touple1[0] == touple2[0]:
        return 0
    return 1


# vraca listu svih likova
def get_all_tags(tagged_list, letter):
    return_list = []
    for (tag, word) in tagged_list:
        word_lower = word.lower()
        word_lower = word_lower.replace(';', '')
        if tag == letter and word_lower not in return_list and word_lower.isalnum() and len(word_lower) > 1:
            return_list = return_list + [word_lower]
    return return_list


# zapisuje u file file_name koja je prica , tagove stroja, nase tagove, i matricu
def write_to_file(file_name, story_num, machine_tag, our_tag, conf_mtr):
    f = open(file_name, 'a')
    f.write(story_num + ':\n')
    f.write('comp tag: ' + ' '.join(machine_tag))
    f.write('\n')
    f.write('our tag: ' + ' '.join(our_tag))
    f.write('\n')
    f.write(np.array_str(conf_mtr))
    f.write('\n\n\n')
    f.close()


# racuna matricu te poziva zapisivanje
def calc(conf_mtr, machine_tag, our_tag, file_name, model_type):
    machine_tag_c_set = get_all_tags(machine_tag, 'C')
    our_tag_c_set = get_all_tags(our_tag, 'C')
    our_tag_c_set_copy = our_tag_c_set[:]
    our_tag_o_set = get_all_tags(our_tag, 'O')

    for word in machine_tag_c_set:
        if word in our_tag_c_set_copy:
            conf_mtr[0][0] += 1
            our_tag_c_set_copy.remove(word)
        else:
            conf_mtr[1][0] += 1

    conf_mtr[0][1] = len(our_tag_c_set_copy)
    # other su oznaceni kao other - oni koji nisu trebali biti other
    conf_mtr[1][1] = len(our_tag_o_set) - conf_mtr[1][0]
    write_to_file(model_type + '.txt', file_name, machine_tag_c_set, our_tag_c_set, conf_mtr)


# funkcija za izradu matrice conf
def make_conf_matrix(conf_matrix, txt_file_for_test, model, tag_model):
    for q in range(0, len(txt_file_for_test)):
        # ne postoje .txt filovi iznad 533 pa sam ovaj uvijet stavio
        if int(txt_file_for_test[q]) > 533:
            continue
        if tag_model == 'crf':
            tag_file_with_crf_model(txt_file_for_test[q], model)
        elif tag_model == 'hmm':
            tag_file_with_hmm_model(txt_file_for_test[q], model)
        nor_tag = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + txt_file_for_test[q] + '_' + tag_model + '.tsv')
        tag_punct = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + txt_file_for_test[q] + '_' + tag_model + '_punct.tsv')
        tag_lower = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + txt_file_for_test[q] + '_' + tag_model + '_lower.tsv')
        tag_lower_punct = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + txt_file_for_test[q] + '_' + tag_model + '_lower_punct.tsv')
        our_tag = parse_tsv(TRAINING_DATA + '/' + str(int(txt_file_for_test[q])) + '.tsv')

        calc(conf_matrix, nor_tag, our_tag, txt_file_for_test[q], tag_model)
        global mtr_nor
        mtr_nor += conf_matrix
        conf_matrix = np.array([[0, 0], [0, 0]])

        calc(conf_matrix, tag_punct, our_tag, txt_file_for_test[q],  tag_model + '_punct')
        global mtr_punct
        mtr_punct += conf_matrix
        conf_matrix = np.array([[0, 0], [0, 0]])

        calc(conf_matrix, tag_lower, our_tag, txt_file_for_test[q], tag_model + '_lower')
        global mtr_lower
        mtr_lower += conf_matrix
        conf_matrix = np.array([[0, 0], [0, 0]])

        calc(conf_matrix, tag_lower_punct, our_tag, txt_file_for_test[q], tag_model + '_lower_punct')
        global mtr_lower_punct
        mtr_lower_punct += conf_matrix
        conf_matrix = np.array([[0, 0], [0, 0]])


def cv(tag_model):
    # dohvacanje sve iz direktorija
    file_list = os.listdir(TRAINING_DATA)
    models = Models()
    # file_for_train = []
    # file_for_test = []
    start = 0
    # matrice da vidimo koliko je koji model dobar

    while start < len(file_list):
        # ako je zadnji dio, pokupi sve ostale , inace uzmi ih 7
        if start + 9 >= len(file_list):
            file_for_test = file_list[start:]
        else:
            file_for_test = file_list[start:start + 8]
        file_for_test_txt = []
        file_for_train = []
        # fileovi za treniranje
        for story in file_list:
            if story not in file_for_test:
                file_for_train = file_for_train + [story]
        # filovi za testiranje samo ime bez .tsv
        for i in range(0, len(file_for_test)):
            file_for_test_txt = file_for_test_txt + [file_for_test[i].replace(".tsv", "")]

        if tag_model == 'crf':
            models.retrain_crf_models(file_for_train)
        elif tag_model == 'hmm':
            models.retrain_hmm_models(file_for_train)
        crf_conf_matrix = np.array([[0, 0], [0, 0]])
        make_conf_matrix(crf_conf_matrix, file_for_test_txt, models,tag_model)
        start += 8


mtr_nor = np.array([[0, 0], [0, 0]])
mtr_punct = np.array([[0, 0], [0, 0]])
mtr_lower = np.array([[0, 0], [0, 0]])
mtr_lower_punct = np.array([[0, 0], [0, 0]])

cv('crf')

print('obicni :\n')
print(mtr_nor)
print('\n\n punct :\n')
print(mtr_punct)
print('\n\n lower :\n')
print(mtr_lower)
print('\n\n lower punct :\n')
print(mtr_lower_punct)
