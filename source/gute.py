import os
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import nltk.data
from nltk import word_tokenize
from nltk.tag import StanfordNERTagger

current_path = os.getcwd() + "/lib/nltk_data"
nltk.data.path.append(current_path)

st = StanfordNERTagger('./lib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       './lib/stanford-ner/stanford-ner.jar',
                       encoding='utf-8')

for i in range(2400, 2600):

    try:
        # read story from www.gutenberg.org/ebooks/i
        text = strip_headers(load_etext(i)).strip()

    except ValueError:
        print("error reading story with id=" + i)
        continue

    tokenized_text = word_tokenize(text)
    tagged_text = st.tag(tokenized_text)
    characters = []
    k = 0
    for entity in tagged_text:
        if entity[1] == 'PERSON':
            if entity[0] not in characters:
                characters.append(entity[0])

    character_string = ', '.join(characters)

    file_name = "data/training/stories/" + str(i) + "_story.txt"
    fo = open(file_name, "w+")
    fo.write(text)
    fo.close()

    file_name = "data/training/characters/" + str(i) + "_characters.txt"
    fo = open(file_name, "w+")
    fo.write(character_string)
    fo.close()
