''' This is the final one'''
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from numpy import size
from nltk.tag import pos_tag
from _collections import defaultdict
from nltk.util import Index
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.chunk.regexp import RegexpParser
from nltk.tree import ParentedTree
import nltk
from nltk.chunk import ne_chunk
from operator import itemgetter




tex='''In 965 AD, Odin, king of Asgard, wages war against the Frost Giants of Jotunheim and their leader Laufey, to prevent them from conquering the nine realms, starting with Earth. The Asgardian warriors defeat the Frost Giants and seize the source of their power, the Casket of Ancient Winters.

In the present,[N 2] Odin's son Thor prepares to ascend to the throne of Asgard, but is interrupted when Frost Giants attempt to retrieve the Casket. Against Odin's order, Thor travels to Jotunheim to confront Laufey, accompanied by his brother Loki, childhood friend Sif and the Warriors Three: Volstagg, Fandral, and Hogun. A battle ensues until Odin intervenes to save the Asgardians, destroying the fragile truce between the two races. For Thor's arrogance, Odin strips his son of his godly power and exiles him to Earth as a mortal, accompanied by his hammer Mjolnir, now protected by an enchantment that allows only the worthy to wield it.

Thor lands in New Mexico, where astrophysicist Dr. Jane Foster, her assistant Darcy Lewis, and mentor Dr. Erik Selvig, find him. The local populace finds Mjolnir, which S.H.I.E.L.D. agent Phil Coulson soon commandeers before forcibly acquiring Jane's data about the wormhole that delivered Thor to Earth. Thor, having discovered Mjolnir's nearby location, seeks to retrieve it from the facility that S.H.I.E.L.D. quickly constructed but he finds himself unable to lift it, and is captured. With Selvig's help, he is freed and resigns himself to exile on Earth as he develops a romance with Jane.

Loki discovers that he is Laufey's biological son, adopted by Odin after the war ended. A weary Odin falls into the deep "Odinsleep" to recover his strength. Loki seizes the throne in Odin's stead and offers Laufey the chance to kill Odin and retrieve the Casket. Sif and the Warriors Three, unhappy with Loki's rule, attempt to return Thor from exile, convincing Heimdall, gatekeeper of the Bifröst—the means of traveling between worlds—to allow them passage to Earth. Aware of their plan, Loki sends the Destroyer, a seemingly indestructible automaton, to pursue them and kill Thor. The warriors find Thor, but the Destroyer attacks and defeats them, prompting Thor to offer himself instead. 

Struck by the Destroyer and near death, Thor's sacrifice proves him worthy to wield Mjolnir. The hammer returns to him, restoring his powers and enabling him to defeat the Destroyer. Kissing Jane goodbye and vowing to return, he and his fellow Asgardians leave to confront Loki.

In Asgard, Loki betrays and kills Laufey, revealing his true plan to use Laufey's attempt on Odin's life as an excuse to destroy Jotunheim with the Bifröst Bridge, thus proving himself worthy to his adoptive father. Thor arrives and fights Loki before destroying the Bifröst Bridge to stop Loki's plan, stranding himself in Asgard. Odin awakens and prevents the brothers from falling into the abyss created in the wake of the bridge's destruction, but Loki allows himself to fall when Odin rejects his pleas for approval. Thor makes amends with Odin, admitting he is not ready to be king; while on Earth, Jane and her team search for a way to open a portal to Asgard.

In a post-credits scene, Selvig has been taken to a S.H.I.E.L.D. facility, where Nick Fury opens a briefcase and asks him to study a mysterious cube-shaped object,[N 3] which Fury says may hold untold power. An invisible Loki prompts Selvig to agree, and he does.'''


 
def norm(word,pos='x'):                 #normalizes all words except proper nouns
    word = word.lower()
    if pos not in ['NNP','NNPS']:
        wnl = WordNetLemmatizer()
        word = wnl.lemmatize(word)
    return (word)     
 
 
sentList = sent_tokenize(tex)   #list of all tokenized sentences
            
print(sentList)            
            
sentNounDict = defaultdict(list)    # a dictionary key:sentence_number value:all nouns in the sentence... (nouns are normalised)

for s in sentList:
    for w,pos in pos_tag(word_tokenize(s)):
        if pos in ['NN','NNS','NNP','NNPS']:
            sentNounDict[sentList.index(s)].append(norm(w,pos))
print (sentNounDict)



wordSentDict = defaultdict(list)    # a dictionary key:(word,pos) value:all sentences it appears in...(word is normalised)

for s in sentList:
    for w,pos in pos_tag(word_tokenize(s)):
        wordSentDict[(norm(w,pos),pos)].append(sentList.index(s))     
print (wordSentDict)


#list of all nouns in the text
listOfNouns = list(sorted(set([norm(w,pos) for s in sentList for w,pos in pos_tag(word_tokenize(s)) if pos in ['NN','NNS','NNP','NNPS']])))
print (listOfNouns)





listOfTaggedSents = []      #list of sentences of tokenized words with postags- list[tuple(w,pos)]

for s in sentList:
    l = [(n,pos) for n,pos in pos_tag(word_tokenize(s))]
    listOfTaggedSents.append(l)
print (listOfTaggedSents) 

mostSigNoun = []                #most recently encountered significant noun
mostSigNounObject =  []         #most recently encountered significant noun which is not a person
mostSigNounPerson = []          #most recently encountered significant noun which has named entity as person

pronounNounDict = defaultdict(list) #key:touple(pronoun,sentence_num) val:list(list(touple(noun,pos)))(noun not normalized)

#grammar for tagging noun phrases and pronouns
grammar =   """NP:{<DT>?<JJ>*(<NN.*>)+}    
               PR:{<PRP.*>}
            """
rp = RegexpParser(grammar)
for s in listOfTaggedSents:
    begin = True
    chunkedTree = ParentedTree.convert(rp.parse(s))         #tree of chunked parts of the sentence
    neTree = ne_chunk(s)                                    #tree with named entity tags
    #print (chunkedTree)
    #chunkedTree.draw()
    for n in chunkedTree:
        if isinstance(n,nltk.tree.Tree):            
            if n.label()=='NP':
                if begin == True:
                    mostSigNoun = [w for w in n if w[1] in ['NN','NNS','NNP','NNPS']]
                    #print (mostSigNoun)
                    for ne in neTree:
                        if isinstance(ne,nltk.tree.Tree):
                            if ne[0] in mostSigNoun:
                                if ne.label() =='PERSON':
                                    mostSigNounPerson = []
                                    mostSigNounPerson.append(ne[0])
                                else:
                                    mostSigNounObject = []
                                    mostSigNounObject.append(ne[0])    
                    begin = False
                
            if n.label()=='PR':
                pron = n[0][0].lower()
                #print pron
                if pron in ['it','its']:    #for objects
                    if len(mostSigNounObject)>0:        
                        pronounNounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNounObject)
                    else:   #if mostsignounobject does not exist
                        pronounNounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNoun)
                else:
                    if len(mostSigNounPerson)>0:
                        pronounNounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNounPerson)
                    else:
                        pronounNounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNoun)    
                begin = False  
                #print pronounNounDict         
                
                #adding the nouns corresponding to the pronouns to sentworddict and wordsentdict
                for v1 in pronounNounDict[(pron,listOfTaggedSents.index(s))]:   
                    for v11 in v1:  #it is a list of lists
                        sentNounDict[listOfTaggedSents.index(s)].append(norm(v11[0],v11[1]))
                        wordSentDict[(norm(v11[0],v11[1]),v11[1])].append(listOfTaggedSents.index(s))
                           
print (sentNounDict)
print (wordSentDict)
print (pronounNounDict)

for key,val in sentNounDict.items():    #making sentnoundict a set
    val = list(set(val))
    sentNounDict[key] = val
print (sentNounDict)


#following code calculates the distance between two phrases
distance = defaultdict(int)             #a dict.. key:(noun or noun(pronoun),sentence_num) value:position in the sentence from the begining
            
for s in listOfTaggedSents:
    dist = 0
    chunkedTree = ParentedTree.convert(rp.parse(s))
    for n in chunkedTree:
        if isinstance(n,nltk.tree.Tree):            
            if n.label()=='NP':
                tempNoun = [w[0] for w in n if w[1] in ['NN','NNS','NNP','NNPS']]
                for w in tempNoun:
                    distance[(norm(w),listOfTaggedSents.index(s))] = dist
            if n.label()=='PR':
                pron = n[0][0].lower()
                tempNoun = pronounNounDict[(pron,listOfTaggedSents.index(s))]                
                for v1 in tempNoun:
                    for v11 in v1:
                        distance[(norm(v11[0],v11[1]),listOfTaggedSents.index(s))] = dist
        dist+=1
print (distance)




#the following code assigns relation factor between two nouns
nounGraph = np.zeros((len(listOfNouns),len(listOfNouns)))

for key,value in sentNounDict.items():
    for v1 in value:
        for v2 in value:
            d=0
            if v2!=v1:
                d = distance[v1,key] - distance[v2,key]
                nounGraph[listOfNouns.index(v1)][listOfNouns.index(v2)] += float((100/(abs(d)+1)))
                #if nounGraph[listOfNouns.index(v1)][listOfNouns.index(v2)]>=100:
                print(v1+' '+v2+" "+str(d))



print(nounGraph)

                
nounPriority = defaultdict(int)             #dict to hold noun priorities... key:noun(normalized)  value:priority
sentencePriority = defaultdict(int)         #dict to hold sentence priorities...key:sentence_num   value:priority


def calcNounPriority():                     #function calculates the noun priority(sum of weights of all the edges attached to this noun in the noungraph)
    total = 0
    i=0
    for x in nounGraph:
        total = sum(x)
        nounPriority[listOfNouns[i]]=total
        i += 1
print (sorted(nounPriority.items(),key=lambda x:x[1], reverse=True))



def calcSentPriority():                     #function calculates sentence priority(sum of priorities of all nouns in the sent)
    for key,value in sentNounDict.items():
        total = 0
        for n in value:
            total += nounPriority[n]
            sentencePriority[key] = total
        



calcNounPriority()
calcSentPriority()



        
print (sorted(sentencePriority.items(),key=lambda x:x[1], reverse=True))
for i in range(len(sentList)):
    print(str(i)+' '+sentList[i])



reducingFactor = 0.9    #10%
summary = []        #list to hold the summary
reduce_per=50/100
print(reduce_per)
for i in range(int(len(sentencePriority)/3)):
    summary.append(max(sentencePriority.items(),key = lambda x:x[1])) 
    print (summary)
    j = summary[-1][0]
    
    for n in sentNounDict[j]:
        nounPriority[n] *= reducingFactor                   #reduce the priority of all nouns in the picked sentence
    
    del sentNounDict[j]         
    del sentencePriority[j]                                #remove the picked sentence
    calcSentPriority()                                      #recalculate sentence priority

print ("\n\n")
i=1
new_list = []
for s in sorted(summary):
    print (i,sentList[s[0]])
    i+=1
    new_list.append(sentList[s[0]])

new_list1 = " ".join(new_list)


import subprocess

def execute_unix(inputcommand):
   p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
   (output, err) = p.communicate()
   return output
c = 'espeak -s 125 -v en+m3 --punct="<characters>" "%s" 2>>/dev/null' % new_list1 
execute_unix(c) 

# write out to wav file 
#b = 'espeak -w temp.wav "%s" 2>>/dev/null' % new_list1 

# speak aloud
#c = 'espeak -ven+f3 -k5 -s140 --punct="<characters>" "%s" 2>>/dev/null' % new_list1 #speak aloud

#execute_unix(b) 
