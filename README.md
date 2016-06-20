# Prepoznavanje likova u dječjim pričama

Projekt u sklopu kolegija Strojno učenje na diplomskom smjeru Računarstvo i matematika, PMF Zagreb

### Autori
Tomislav Horina
Gorana Levačić

### Zahtjevi
```
Python ≥ 3.4
gcc
gcc-c++
```

Potrebne Python biblioteke su navedene u requirements.txt i instaliraju se s:
``` sudo pip3 install -r requirements.txt ```

### Upute

U direktoriju `/data/gutenberg-files` se nalaze dokumenti s Project Gutenberga u kojima su ručno
uklonjeni početak do prve priče, tj. header, uvod, sadržaj i sl.) i kraj (od kraja posljednje
priče do kraja čitavog dokumenta).

Skripta `generate_test_data.py` uzima svaki od dokumenata iz gorneg direktorija i piše ga u jedan
privremeni dokument. Sadržaj tog dokumenta se potom čisti od većine nepotrebnih dijelova (opisi
ilustracija, fusnote i sl.), te se dodaju razmaci između interpunkcijskih znakova. Potom se svaka
od priča iz privremenog dokumenta odvaja i sprema u zaseban dokument u direktorij `/data/stories`.
Te priče (ili bilo koji drugi tekst) se može koristiti za testiranje modela.

U istoj skripti se provodi i generiranje tsv dokumenata za svaku od priča. Dokumenti se sastoje od
dva stupca. U prvom stupcu je oznaka 'O', u drugom stupcu je token (riječ ili interpunkcijski znak(ovi)).
Oznaka 'O' označava da je taj token 'other', odnosno nije lik (character). Generirani tsv dokumenti
se pohranjuju u `/data/generated-tsv-files`.

U direktoriju `/data/correctly-tagged-tsv-files` se nalaze neki od generiranih tsv dokumenata u kojima
su ručno promijenjene oznake u 'C' za svaku riječ (ili skup riječi) koja predstavlja lika u toj priči.
Ti dokumenti se kasnije koriste za treniranje modela.

U skripti `models.py` se treniraju modeli HMM (hidden Markov model, skriveni Markovljev mode) i CRF
(conditional random fields, uvjetna slučajna polja). Oba modela se treniraju na četiri načina:
1. da uzimaju interpunkcijske znakove u obzir, te da treniraju nad izvornim oblicima riječi
2. da ne uzimaju interpunkcijske znakove u obzir, te da treniraju nad izvornim oblicima riječi
3. da uzimaju interpunkcijske znakove u obzir, te da treniraju nad lowercase oblicima riječi
4. da ne uzimaju interpunkcijske znakove u obzir, te da treniraju nad lowercase oblicima riječi

Nad HMM i CRF modelima se provodi unakrsna validacija. Skup za validaciju je veličine 3.
Modeli se evaluiraju pomoću F<sub>2</sub>-mjere. Parametar _beta_ je veličine 2 jer
želimo dati veću važnost osjetljivosti testa, odnosno važnije nam je da otkrijemo više likova nego
koliko tih likova je točno. Nakon unakrsne validacije u `/data/cross-validation` se generiraju grafovi
koji opisuju F<sub>2</sub>-mjeru u ovisnosti o iteracijama.

Stanford NER se trenira posebno, s obzirom da NLTK ima samo podršku za označavanje korištenjem već
postojećih modela. Iz tog razloga se ne provodi unakrsna validacija.
U `/lib/stanford-ner` se nalazi `properties` datoteka u kojoj su navedene postavke za treniranje novog modela. 
Koriste se defaultne postavke.
Novi model se trenira pokretanjem sljedećih naredbi iz tog direktorija:
```
java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier -prop ner.properties -trainFile training-sets/tokenized_content.tsv -serializeTo classifiers/trained_stanford_ner.ser.gz
java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier -prop ner.properties -trainFile training-sets/tokenized_content_lower.tsv -serializeTo classifiers/trained_stanford_ner_lower.ser.gz
java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier -prop ner.properties -trainFile training-sets/tokenized_content_punct.tsv -serializeTo classifiers/trained_stanford_ner_punct.ser.gz
java -mx4g -cp ".*:lib/*:stanford-ner.jar" edu.stanford.nlp.ie.crf.CRFClassifier -prop ner.properties -trainFile training-sets/tokenized_content_lower_punct.tsv -serializeTo classifiers/trained_stanford_ner_lower_punct.ser.gz
```

`mx4g` parametar zadaje 4gb radne memorije za ovaj proces. Ako Java javi grešku pri treniranju modela, potrebno je 
pokrenuti ponovno proces s s manje memorije. Detaljnije piše [ovdje](http://nlp.stanford.edu/software/crf-faq.shtml#d).
__Napomena__: Ukoliko se dogodi da treniranje prestane, a Python proces je također prekinut, moguće je do kraja dovršiti 
treniranje ova 4 modela preko konzole, bez ponovnog treniranja HMM i CRF modela. Pri ponovnom pokretanju Python procesa 
potrebno je odabrati `n`, odnosno učitavanje već istreniranih modela.

Prilikom pokretanja skripte se može odabrati hoće li se koristiti već istrenirani modeli ili će se
trenirati novi modeli. Ukoliko želimo koristiti već istrenirane modele, a oni ne postoje, pokrenut će se novo
treniranje modela. Nakon treniranja ili učitavanja modela, modeli se testiraju nad podacima iz `/data/test/stories` i 
uspoređuju s ručno označenim podacima iz `/data/our_tag`. Potom se u `/data/test/results` generiraju dokumenti s 
detaljnijim izvješćem o svakoj od metoda.

Nakon treniranja novih modela ili učitavanja već postojećih, korisnik može testirati rad svih modela tako da unese naziv
priče iz direktorija `/data/stories` koju želi označiti. Na izlaz će se ispisati rezultati svake od metoda, te će se u
`/data/test/tagged` generirati .tsv dokumenti u kojima se vidi koje riječi su označene kojom oznakom.