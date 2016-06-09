import csv
import os
import numpy as np
import re
from collections import Counter

from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models
from source.utils import write_tagged_content_to_file
from source.labeled_tokens import part_populate_labeled_tokens
TRANING_DATA = '../data/training-data'
TEST_FILES_PATH = '../data/test-files/stories'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'
ORGINAL_STORIES = '../data/stories/'


# neka od tvojih funkcija iz train_and_test_models.py
def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
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
    path = ORGINAL_STORIES + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File ' + path + ' does not exist!')
        return

    content = get_content(path)
    tokenized_content = wordpunct_tokenize(content)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, model.crf, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
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


# funkcija za izradu matrice conf
def make_conf_matrix(conf_matrix, txt_file_for_test, model):
    for q in range(0, len(txt_file_for_test)):
        # ne postoje .txt filovi iznad 533 pa sam ovaj uvijet stavio
        if int(txt_file_for_test[q]) > 533:
            continue

        tag_file_with_crf_model(txt_file_for_test[q], model)
        crf_tag = parse_tsv(TAGGED_TEST_FILES_PATH + '/' + txt_file_for_test[q] + '_crf.tsv')
        our_tag = parse_tsv(TRANING_DATA + '/'+str(int(txt_file_for_test[q])) + '.tsv')

        # print(file_for_test_txt[i])
        crf_tag_c_set = get_all_tags(crf_tag, 'C')
        our_tag_c_set = get_all_tags(our_tag, 'C')
        our_tag_c_set_copy = our_tag_c_set[:]
        our_tag_o_set = get_all_tags(our_tag, 'O')

        for word in crf_tag_c_set:
            if word in our_tag_c_set_copy:
                conf_matrix[0][0] += 1
                our_tag_c_set_copy.remove(word)
            else:
                conf_matrix[1][0] += 1
        print(crf_tag_c_set)
        print(our_tag_c_set)
        conf_matrix[0][1] = len(our_tag_c_set_copy)
        # other su oznaceni kao other - oni koji nisu trebali biti other
        conf_matrix[1][1] = len(our_tag_o_set) - conf_matrix[1][0]
        print(conf_matrix)
        conf_matrix = np.array([[0, 0], [0, 0]])

        '''
        minimum = min(len(crf_tag),len(our_tag))
        for j in range(0,minimum):
            if( compare(crf_tag[j], our_tag[j]) == 0 ):
                if ( crf_tag[j][0] == 'C' ):
                    conf_matrix[0][0]= conf_matrix[0][0] + 1
                else:
                    conf_matrix[1][1] = conf_matrix[1][1] + 1
            else:
                if (crf_tag[j][0] == 'C'):
                    conf_matrix[1][0] = conf_matrix[1][0] + 1
                else:
                    conf_matrix[0][1] = conf_matrix[0][1] + 1
        '''


# dohvacanje sve iz direktorija
file_list = os.listdir(TRANING_DATA)

file_for_train = []
file_for_test = []
i = 0
# raspodjela na train and test omjer 0,7:0,3
for file in file_list:
    if 0.7 * len(file_list) > i:
        file_for_train = file_for_train + [file]
        i += 1
    else:
        file_for_test = file_for_test + [file]

print(file_for_test)
# dohvacanje svih imena tsvova
file_for_test_txt = []
for i in range(0, len(file_for_test)):
    file_for_test_txt = file_for_test_txt + [file_for_test[i].replace(".tsv", "")]

print(file_for_test_txt)

# kreiranje testa za treniranje
# trenutno bespotrebno
labeled_test = []
for j in file_for_test:
    labeled_test = labeled_test + part_populate_labeled_tokens([str(j)], False, False)

crf_conf_matrix = np.array([[0, 0], [0, 0]])
models = Models()
# models.retrain_part_models([])
# y_crf_data = [models.crf.evaluate([labeled_test])]
# make_conf_matrix(crf_conf_matrix,file_for_test_txt,models)
# y_hmm_data = [models.hmm.test([labeled_test])]
x_data = []  # trenutno bespotrebno

# print(crf_conf_matrix)
# crf_conf_matrix = np.array([[0,0],[0,0]])
i = 0
list_dir = []
# ovo je trebalo biti da se iterativno uci tj , prvo uci na 5 pa na 10 itd.
for j in file_for_train:
    if i > -1:
        models.retrain_part_models(file_for_train)
        # y_crf_data = y_crf_data + [models.crf.evaluate([labeled_test])]
        # y_hmm_data = y_hmm_data + [models.hmm.evaluate([labeled_test])]
        make_conf_matrix(crf_conf_matrix, file_for_test_txt, models)
        # print(crf_conf_matrix)
        # crf_conf_matrix = np.array([[0, 0], [0, 0]])
        # x_data = x_data + [j]
        i = 0
        break
        # list_dir = []

    list_dir = list_dir + [j]
    i += 1

# models.retrain_models()
# y_crf_data =y_crf_data + [models.crf.evaluate([labeled_test])]
# y_hmm_data = y_hmm_data + [models.hmm.test([labeled_test])]
# make_conf_matrix(crf_conf_matrix, file_for_test_txt, models)
# print(crf_conf_matrix)

# print (y_crf_data)
# print (y_hmm_data)
