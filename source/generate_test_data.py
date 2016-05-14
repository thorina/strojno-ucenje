import os
import re
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
regex_tag = re.compile('\[\d+\]')
regex_songs = re.compile('\"\n')
regex_footnotes = re.compile('(footnote)s?', flags=re.IGNORECASE)
regex_footnote = re.compile('\[(footnote)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]+\]', flags=re.IGNORECASE)
regex_illustration = re.compile('\n*\[(illustration)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]*\]', flags=re.IGNORECASE)

regex_story_separator = \
    re.compile('\n{3,4}[\w \t:_$&%"\',\-\.\?!\(\)\\\/]+(\n{1,2}[\w \t:_$&%"\',\-\.\?!\(\)\\\/]*){1,2}\n{2,4}')

path = '../data/gutenberg_stripped_files'
cnt = 0

for filename in os.listdir(path):

    file = open(path + '/' + filename, 'r')
    file_content = file.read()
    file.close()

    file_content = regex_extra_whitespaces.sub(' ', file_content)
    file_content = regex_songs.sub('', file_content)
    file_content = regex_tag.sub('', file_content)
    file_content = regex_footnote.sub('', file_content)
    file_content = regex_footnotes.sub('', file_content)
    file_content = regex_illustration.sub('', file_content)
    separated_stories = regex_story_separator.split(file_content)

    for story in separated_stories:

        # regex_story_separator has () so split will return short or empty string that are part of separator
        # from documentation:
        # If capturing parentheses are used in pattern, then the text of all groups
        # in the pattern are also returned as part of the resulting list.
        if len(story) < 10:
            continue

        cnt += 1
        file_name = '../data/training/stories/' + str(cnt) + '_story.txt'
        file = open(file_name, 'w+')
        file.write(story)
        if len(story) < 300:
            print(str(cnt) + " - " + str(len(story)))

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
