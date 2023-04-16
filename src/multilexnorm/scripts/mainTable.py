import sys
import os
import numpy as np
from optparse import OptionParser
from os import listdir
from os.path import isfile, join

def err(msg):
    print('Error: ' + msg)
    exit(0)

def evaluate(raw, gold, pred, ignCaps=False, verbose=False):
    cor = 0
    changed = 0
    total = 0

    if len(gold) != len(pred):
        print('Error: gold normalization contains a different numer of sentences(' + str(len(gold)) + ') compared to system output(' + str(len(pred)) + ')')
        exit(1)

    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            err('Error: a sentence has a different length in you output, check the order of the sentences')
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            if ignCaps:
                wordRaw = wordRaw.lower()
                wordGold = wordGold.lower()
                wordPred = wordPred.lower()
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            elif verbose:
                print(wordRaw, wordGold, wordPred)
            total += 1

    accuracy = float(cor) / total
    lai = float(total - changed) / total
    err = float(accuracy - lai) / (1-lai)
    if verbose:
        print(cor, changed, total)

        print('Baseline acc.(LAI): {:.2f}'.format(lai * 100)) 
        print('Accuracy:           {:.2f}'.format(accuracy * 100)) 
        print('ERR:                {:.2f}'.format(err * 100))

    return lai, accuracy, err



def sanity_check(rawData, predData, info='', verbose=False):
    assert len(rawData) == len(predData)
    if verbose>1:
        print(str(len(rawData)) + ' number of sentences have been loaded (raw and predicted data to be parsed)')
    n_sanity_checked  = 0
    for ind, (sentRaw, sentPred) in enumerate(zip(rawData, predData)):
        assert len(sentRaw) == len(sentPred), str(ind) + ' raw sentence not aligned with predicted one '
        for wordRaw, wordPred in zip(sentRaw, sentPred):
            assert wordRaw == wordPred, wordRaw + ' <> ' + wordPred
            n_sanity_checked += 1
    if verbose>1:
        print(str(n_sanity_checked) + ' token have been sanity checked')
    if verbose:
        print('Predicted '+submission_answer_file+ ' data ready to be parsed for extrinsic evaluation')
    


def loadRawData(path):
    rawData = []
    curSent = []
    
    for line in open(path):
        tok = line.strip()
        if len(tok.split('\t')) > 1:
            err('erroneous input, line:\n' + line + '\n in file ' + path + ' contains more then 1 element')
        if len(tok) == 0:
            rawData.append([x for x in curSent])
            curSent = []
        else:
            curSent.append(tok)

    # in case file does not end with newline
    if curSent != []:
        rawData.append([x for x in curSent])
    
    return rawData


def loadNormData(path):
    rawData = []
    goldData = []
    curSent = []

    for line in open(path):
        tok = line.strip().split('\t')

        if tok == [''] or tok == []:
            rawData.append([x[0] for x in curSent])
            goldData.append([x[1] for x in curSent])
            curSent = []
        else:
            if len(tok) > 2:
                err('erroneous input, line:\n' + line + '\n in file ' + path + ' contains more then two elements')
            if len(tok) == 1:
                tok.append('')
            curSent.append(tok)

    # in case file does not end with newline
    if curSent != []:
        rawData.append([x[0] for x in curSent])
        goldData.append([x[1] for x in curSent])
    return rawData, goldData


def getScores(teamName):
    scores = []
    for lang in sorted(os.listdir('data')):
        truth_file = 'data/' + lang + '/test.norm'
        answer_file = 'submissions/' + teamName + '/intrinsic_evaluation/' + lang + '/test.norm.pred'
        goldRaw, goldNorm = loadNormData(truth_file)
        predRaw, predNorm = loadNormData(answer_file)

        ignCaps = False if lang in ['da', 'de', 'it', 'nl', 'tr', 'trde'] else True 
    
        lai, accuracy, err = evaluate(goldRaw, goldNorm, predNorm, ignCaps, False)
        scores.append(err)
    return [sum(scores)/len(scores)] + scores + [teamName]

if __name__ == '__main__':
    data = [] 
    allScores = []
    for team in os.listdir('submissions'):
        print(team)
        scores = getScores(team)
        allScores.append(scores)
    for team in sorted(allScores, reverse=True):
        add = ''
        if 'thunder' in team[-1] or 'Dives' in team[-1]:
            team[-1] += '$^*$'
        data.append(team[-1:] + team[:-1])

for langIdx in range(1, len(data[0])):
    highest = 0.0
    for teamIndex in range(len(data)):
        if data[teamIndex][langIdx] > highest:
            highest = data[teamIndex][langIdx]
    for teamIndex in range(len(data)):
        if data[teamIndex][langIdx] == highest:
            data[teamIndex][langIdx] = '\\textbf{' + '{:.2f}'.format(data[teamIndex][langIdx]*100) + '}'
        else:
            data[teamIndex][langIdx] = '{:.2f}'.format(data[teamIndex][langIdx]*100)

print()    
print(' & '.join(['team', 'macro-avg.'] + sorted(os.listdir('data'))) + ' \\\\')
for row in data:
    print(' & '.join(row) + ' \\\\')



