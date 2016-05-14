import os
import re
import tempfile
import nltk.data
# from nltk import word_tokenize
from nltk.tag import StanfordNERTagger

# current working directory = source
current_path = os.getcwd() + '/lib/nltk_data'
nltk.data.path.append(current_path)

st = StanfordNERTagger('../lib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '../lib/stanford-ner/stanford-ner.jar',
                       encoding='utf-8')

regex_extra_whitespaces = re.compile('( ){2,}')
regex_footnotes = re.compile('(footnote)s?', flags=re.IGNORECASE)
regex_footnote = re.compile('\[(footnote)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]+\]', flags=re.IGNORECASE)
regex_illustration = re.compile('\n*\[(illustration)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]*\]', flags=re.IGNORECASE)

path = '../data/gutenberg_stripped_files'

with tempfile.NamedTemporaryFile(mode='w+t') as temp:
    for filename in os.listdir(path):
        file_path = path + '/' + filename
        with open(file_path, 'r') as file:
            content = file.read()
            content = regex_extra_whitespaces.sub(' ', content)
            content = regex_footnote.sub('', content)
            content = regex_footnotes.sub('', content)
            content = regex_illustration.sub('', content)
            temp.write(content)

    # story separators are usually in this form:
    # 2-3 empty lines
    # 1 short line title or chapter number
    # optionally: 1-2 empty lines + 1 short line title or author
    # 2-3 empty lines

    story_count = 0
    file_name = '../data/training/stories/' + str(story_count) + '_story.txt'
    file = open(file_name, 'w+')

    temp.seek(0)
    lines = temp.readlines()
    last_index = len(lines)-6
    i = 0
    while i < last_index:
        before_title = lines[i].isspace() and lines[i + 1].isspace()
        title = not lines[i + 2].isspace() and len(lines[i + 2]) < 50
        blank_line_after_title = lines[i + 3].isspace()
        after_title_variant_one = lines[i + 4].isspace()
        after_title_variant_two = not lines[i + 4].isspace() and len(lines[i + 4]) < 60 \
                                  and lines[i + 5].isspace() \
                                  and lines[i + 6].isspace()

        if before_title and title and blank_line_after_title and (after_title_variant_one or after_title_variant_two):
            story_count += 1
            print("story count: " + str(story_count))
            file.close()
            file_name = '../data/training/stories/' + str(story_count) + '_story.txt'
            file = open(file_name, 'w+')
            if after_title_variant_one:
                i += 5
            else:
                i += 7
        else:
            file.write(lines[i])
            i += 1

    for i in range(last_index, len(lines)):
        file.write(lines[i])

# tokenized_text = word_tokenize(text)
# tagged_text = st.tag(tokenized_text)
# characters = []
# k = 0
# for entity in tagged_text:
#     if entity[1] == 'PERSON':
#         if entity[0] not in characters:
#             characters.append(entity[0])
#
# character_string = ', '.join(characters)
#
# file_name = "data/training/stories/" + str(i) + "_story.txt"
# fo = open(file_name, "w+")
# fo.write(text)
# fo.close()
#
# file_name = "data/training/characters/" + str(i) + "_characters.txt"
# fo = open(file_name, "w+")
# fo.write(character_string)
# fo.close()
