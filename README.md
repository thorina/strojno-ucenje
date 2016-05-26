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
``` sudo pip3.5 install -r requirements.txt ```

### Upute

U direktoriju `/data/gutenberg-files` se nalaze dokumenti s Project Gutenberga u kojima su ručno
uklonjeni početak do prve priče, tj. header, uvod, sadržaj i sl.) i kraj (od kraja posljednje
priče do kraja čitavog dokumenta).

Skripta `generate_test_data.py` uzima svaki od dokumenata iz gorneg direktorija i piše ga u jedan
privremeni dokument. Sadržaj tog dokumenta se potom čisti od većine nepotrebnih dijelova (opisi
ilustracija, fusnote i sl.), te se dodaju razmaci između interpunkcijskih znakova. Potom se svaka
od priča iz privremenog dokumenta odvaja i sprema u zaseban dokument u direktorij `\data\stories`.
Te priče (ili bilo koji drugi tekst) se može koristiti za testiranje modela.

U istoj skripti se provodi i generiranje tsv dokumenata za svaku od priča. Dokumenti se sastoje od
dva stupca. U prvom stupcu je oznaka 'O', u drugom stupcu je token (riječ ili interpunkcijski znak(ovi)).
Oznaka 'O' označava da je taj token 'other', odnosno nije lik (character). Generirani tsv dokumenti
se pohranjuju u `/data/generated-tsv-files`.

U direktoriju `/data/correctly-tagged-tsv-files` se nalaze neki od generiranih tsv dokumenata u kojima
su ručno promijenjene oznake u 'C' za svaku riječ (ili skup riječi) koja predstavlja lika u toj priči.
Ti dokumenti se kasnije koriste za treniranje modela.

U skripti `models.py` se treniraju modeli HMM (hidden Markov model, skriveni Markovljev mode) i CRF
(conditional random fields, uvjetna slučajna polja). Oba modela se treniraju na dva načina - da uzimaju
interpunkcijske znakove u obzir, te da ih ne uzimaju.
Prilikom pokretanja skripte se može odabrati hoće li se koristiti već istrenirani modeli ili će se
trenirati novi modeli.
