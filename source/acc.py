import csv
import os
import numpy as np
import re
from collections import Counter

from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models
from source.utils import write_tagged_content_to_file
from source.labeled_tokens import part_populate_labeled_tokens
TRANING_DATA='../data/training-data'
TEST_FILES_PATH = '../data/test-files/stories'
TAGGED_TEST_FILES_PATH = '../data/test-files/tagged-test-files'
ORGINAL_STORIES = '../data/stories/'


def get_content(path):
    with open(path, 'r') as file_output:
        content = file_output.read()
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
    return content

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

    #counter = Counter(characters)
    #print('Found characters and number of their occurrences:')
    #for character, occurrences in counter.items():
        #print(character + ' ' + str(occurrences))

    return tagged_content

def tag_file_with_crf_model(file_name, models):
    path = ORGINAL_STORIES + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File ' + path + ' does not exist!')
        return

    content = get_content(path)
    tokenized_content = wordpunct_tokenize(content)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.crf, False)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path)



file_list = os.listdir(TRANING_DATA)

file_for_train = []
file_for_test = []
i=0
for file in file_list:
    if 0.7 * len(file_list) > i :
        file_for_train = file_for_train + [file]
        i = i + 1
    else :
        file_for_test = file_for_test + [file]

print (file_for_test)
file_for_test_txt = []
for i in range (0,len(file_for_test)):
    file_for_test_txt =  file_for_test_txt + [file_for_test[i].replace(".tsv","")]

print(file_for_test_txt)

labeled_test = []
for j in file_for_test :
    labeled_test = labeled_test + part_populate_labeled_tokens([str(j)],False, False)

crf_conf_matrix = np.array([[0,0],[0,0]])
models = Models()
models.retrain_part_models([])
y_crf_data = [models.crf.evaluate([labeled_test])]

def parse_tsv(file):
    return_list = []
    with open(file) as reader:
        read = csv.reader(reader,dialect="excel-tab")
        for row in read:
            return_list = return_list + [(row[0],row[1])]
    return return_list

def compare(touple1,touple2):
    if (touple1[0] == touple2[0]):
        return 0
    return 2

def make_conf_matrix(conf_matrix,file_for_test_txt,models):
    for i in range(0,len(file_for_test_txt)):
        if(int(file_for_test_txt[i]) > 533 ):
            continue
        tag_file_with_crf_model(file_for_test_txt[i],models)
        crf_tag = parse_tsv(TAGGED_TEST_FILES_PATH+ '/'+ file_for_test_txt[i] +  '_crf.tsv')
        our_tag = parse_tsv(TRANING_DATA + '/'+str(int(file_for_test_txt[i])) +  '.tsv' )
        print(file_for_test_txt[i])
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


make_conf_matrix(crf_conf_matrix,file_for_test_txt,models)
#y_hmm_data = [models.hmm.test([labeled_test])]
x_data = []

print(crf_conf_matrix)
crf_conf_matrix = np.array([[0,0],[0,0]])
i = 0
list_dir = []
for j in file_for_train:
    if i > 5 :
        models.retrain_part_models(list_dir)
        y_crf_data = y_crf_data + [models.crf.evaluate([labeled_test])]
        #y_hmm_data = y_hmm_data + [models.hmm.evaluate([labeled_test])]
        make_conf_matrix(crf_conf_matrix, file_for_test_txt, models)
        print(crf_conf_matrix)
        crf_conf_matrix = np.array([[0, 0], [0, 0]])
        x_data = x_data + [j]
        i = 0
        #list_dir = []

    list_dir = list_dir + [j]
    i = i + 1

models.retrain_models()
y_crf_data =y_crf_data + [models.crf.evaluate([labeled_test])]
#y_hmm_data = y_hmm_data + [models.hmm.test([labeled_test])]
make_conf_matrix(crf_conf_matrix, file_for_test_txt, models)
print(crf_conf_matrix)

print (y_crf_data)
#print (y_hmm_data)