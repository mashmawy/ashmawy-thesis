import os

tgts=['ud-de-tweede', 'ud-en-aae', 'ud-en-monoise', 'ud-en-tweebank2', 'ud-it-postwita', 'ud-it-twittiro', 'ud-tr-iwt151']

lang2model = {'de':'UD_German-GSD', 'en':'UD_English-EWT', 'it':'UD_Italian-ISDT', 'tr':'UD_Turkish-IMST'}

def getModel(name):
    modelDir = 'machamp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.tar.gz'
            if os.path.isfile(modelPath):
                return modelPath[modelPath.find('/')+1:]
    return ''

def txt2conll(txtPath, conllPath):
    outFile = open(conllPath, 'w')
    lines = [x.strip('\n') for x in open(txtPath).readlines()]
    # Here, we undo merges, as they are non-trivial to evaluate 
    for endIdx in reversed(list(range(1,len(lines)-1))):
        if len(lines[endIdx]) > 0:
            if lines[endIdx][-1] == '\t':
                for length in range(10):
                    if endIdx-length < 0 or (len(lines[endIdx-length]) > 0 and lines[endIdx-length][-1] != '\t'):
                        break
                begIdx = endIdx-length
                for fixIdx in range(begIdx, begIdx + length+1):
                    if len(lines[fixIdx]) == 0:
                        break
                    lines[fixIdx] = lines[fixIdx].split('\t')[0] + '\t' + lines[fixIdx].split('\t')[0]
 
    for word in lines:
        word = word.strip('\n')
        if word == '':
            outFile.write('\n')
        else:
            tok = word.split('\t')
            for word in tok[1].split(' '):
                outFile.write('\t'.join(['_', word] + ['_'] * 4 + ['0','_', '_', '_']) + '\n')
    outFile.close()
            

for team in os.listdir('../submissions'):
    for tgt in tgts:
        normPath = '../submissions/' + team + '/extrinsic_evaluation/' + tgt + '.test.norm.pred'
        lang = tgt.split('-')[1]
        inPath = normPath.replace('pred', 'conllu')
        txt2conll(normPath, inPath)
        for seed in ['1', '2', '3']:
            outPath = normPath.replace('pred', seed + '.conllu.out')
            modelName = lang2model[lang] + '.' + seed
            modelPath = getModel(modelName).replace('machamp/', '')
            cmd = 'python3 predict.py ' + modelPath + ' ../' + inPath + ' ../' + outPath
            print(cmd)
    
