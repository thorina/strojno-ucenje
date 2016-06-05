import csv
import os
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

labeled_test = []
for j in file_for_test :
    labeled_test = labeled_test + part_populate_labeled_tokens([str(j)],False, False)

models = Models()
models.retrain_part_models([])
y_crf_data = [models.crf.evaluate([labeled_test])]
y_hmm_data = [models.hmm.test([labeled_test])]
x_data = []

i = 0
list_dir = []
for j in file_for_train:
    if i > 5 :
        models.retrain_part_models(list_dir)
        y_crf_data = y_crf_data + [models.crf.evaluate([labeled_test])]
        y_hmm_data = y_hmm_data + [models.hmm.test([labeled_test])]
        x_data = x_data + [j]
        i = 0
        list_dir = []

    list_dir = list_dir + [j]
    i = i + 1

print (y_crf_data)
print (y_hmm_data)