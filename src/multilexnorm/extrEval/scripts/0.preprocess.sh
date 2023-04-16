git clone https://github.com/machamp-nlp/machamp.git
cd machamp
git reset --hard 344c3e225025e1269430d4f714ead53a1a44b917
cd ../

cp -r treebanks treebanks-cleaned
python3 machamp/scripts/cleanconll.py treebanks-cleaned/*
sed -i "s;ATMENTION;@username;g" treebanks-cleaned/*

grep -v "^#" treebanks-cleaned/ud-de-tweede.test.conllu | cut -f 2 > treebanks-cleaned/ud-de-tweede.test.txt
grep -v "^#" treebanks-cleaned/ud-en-aae.test.conllu | cut -f 2 > treebanks-cleaned/ud-en-aae.test.txt
grep -v "^#" treebanks-cleaned/ud-en-monoise.test.conllu | cut -f 2 > treebanks-cleaned/ud-en-monoise.test.txt
grep -v "^#" treebanks-cleaned/ud-en-tweebank2.test.conllu | cut -f 2 > treebanks-cleaned/ud-en-tweebank2.test.txt
grep -v "^#" treebanks-cleaned/ud-it-postwita.test.conllu | cut -f 2 > treebanks-cleaned/ud-it-postwita.test.txt
grep -v "^#" treebanks-cleaned/ud-it.twittiro.test.conllu | cut -f 2 > treebanks-cleaned/ud-it.twittiro.test.txt
#grep -v "^#" treebanks-cleaned/ud-tr-iwt151.test.conllu | cut -f 2 > treebanks-cleaned/ud-tr-iwt151.test.txt


