cd machamp

# get data:
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687/ud-treebanks-v2.8.tgz
tar -zxvf ud-treebanks-v2.8.tgz
mv ud-treebanks-v2.8 data
cp -r data/ud-treebanks-v2.8 data/ud-treebanks-v2.8.noEUD
python3 scripts/misc/cleanconl.py data/ud-treebanks-v2.8.noEUD/*/*conllu

# train models
python3 scripts/2.ud.train.py | egrep "German-GSD|Turkish-IMST|Italian-ISDT|English-EWT" > 2.train.sh
chmod +x 2.train.sh
./2.train.sh


