from nltk.tag import StanfordNERTagger

st = StanfordNERTagger('../lib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '../lib/stanford-ner/stanford-ner.jar',
                       encoding='utf-8')