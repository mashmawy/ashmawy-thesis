
testLabels =[] 
testResults =[]

def split(word): 
    return [char for char in word]  
 


def TestFilter(sentence, label): 
    if(sentence.lower() != label.lower()): 
        testLabels.append(label+ " oov")
    elif( "#" not in label and "http" not in label and "@" not in label): 
        testLabels.append(label + " iv")  



datafile = open('output.txt', 'r')
Lines = datafile.readlines()
datafile.close()
 
i2 = 0
while i2 < len(Lines):
    words=  Lines[i2].strip().replace('\n','').replace('\t',' \t ').split('\t')
    if(len(words)> 1):
        testResults.append(str(words[0]).strip().lower()) 
        testLabels.append(str(words[1]).strip().lower()) 
    else:
        testResults.append(str(words[0]).strip().lower()) 
        testLabels.append(str(words[0]).strip().lower()) 
    i2 += 1    
 
tp=0
tn=0
fp=0
fn=0
t=0
i  = 0
baseline=0
while i < len(testLabels): 
    if(testLabels[i]  == testResults[i] ):
            t=t+1  
    elif(' iv' in testLabels[i]  and ' iv' in testResults[i] ):
            t=t+1
    if(' oov' in testLabels[i] and ' oov' in testResults[i] ):
        tp=tp+1
    elif (' iv' in testLabels[i] and ' oov' in testResults[i] ):
        fp=fp+1 
    elif (' oov' in testLabels[i] and ' iv' in testResults[i] ):
        fn=fn+1
    
    if(' oov' in testLabels[i]):
        baseline=baseline+1
    i=i+1
baseline= (baseline/len(testLabels))
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1=2 *(recall * precision) / (recall + precision)
accurecy = t/len(testLabels)

print("accurecy")
print(accurecy)

print("recall")
print(recall)
print("precision")
print(precision)
print("f1")
print(f1)
err =( accurecy-baseline)/(1.0-baseline)
print("err") 
print(err) 