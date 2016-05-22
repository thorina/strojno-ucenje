import csv
import nltk
from nltk.corpus import names

labeled_names = ([(name, 'C') for name in names.words('male.txt')] + [(name, 'C') for name in names.words('female.txt')])

list_story=[] + labeled_names
for i in range (1,30):
    story = []
    with open("data/training-data/" + str( i ) + ".tsv") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            #print ("prica  " + str( i ) + "  -->"  + str( line))
            if (line[1].isalnum()):
                y = line[0]
                x = line[1]
                story = story + [ (x, y) ]
        list_story = list_story + story

symbols = list(set([ss[0] for sss in list_story for ss in sss]))
states = ["O","C"]
hmm_= nltk.tag.hmm.HiddenMarkovModelTrainer(states=states,symbols=symbols)
hmm__ = hmm_.train_supervised( [list_story])


print(hmm__.tag(["king"]))
