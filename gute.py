from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/usr/share/stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

for i in range(2400,2600):
	try:
		text = strip_headers(load_etext(i)).strip()
	except ValueError:
		print ("Greska: nema ",i)
		continue
	tokenized_text = word_tokenize(text)
	classified_text = st.tag(tokenized_text)
	l=[]
	k=0
	for x in classified_text:
		if(x[1]== 'PERSON'):
			kont=False
			for j in range(len(l)):
				if(l[j]==x[0]):
					kont=True
					break
			if(kont == False):
				l.insert(len(l),x[0])

	str1=', '.join(l)
	ime_file="tekstovi/"+str(i)+".txt"
	fo = open(ime_file, "w+")
	fo.write(text)
	fo.close()

	ime_file="likovi/"+str(i)+"_lik.txt"
	fo = open(ime_file, "w+")
	fo.write(str1)
	fo.close()