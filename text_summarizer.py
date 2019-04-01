# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:59:46 2019

@author: Shravan
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from collections import defaultdict
from nltk.chunk.regexp import RegexpParser	

from nltk.tag import pos_tag
#POS tagging is a supervised learning solution that uses features like the previous word, next word, is first letter capitalized and so on.
#The function will load a pretrained tagger from a file. You can see the file name with nltk.tag._POS_TAGGER

from nltk.stem.wordnet import WordNetLemmatizer         
#WordNet has large data (words) stored in lexical structure (synset, lemmas..) it lemmatizes based on the pos_tag provided (default is noun).   
			
from nltk.tree import ParentedTree	
#ParentedTree is a subclass of Tree that automatically maintains parent pointers for single-parented trees.

from nltk.chunk import ne_chunk							
#to chunk only named-entity tags (or to select only required tags)
#ne_chunk contains a pretrained model to classify words to name entities. 
#Name Entities Available in that model are:
#ORGANIZATION	Georgia-Pacific Corp., WHO
#PERSON	        Eddy Bonte, President Obama
#LOCATION	    Murray River, Mount Everest
#DATE	        June, 2008-06-29
#TIME	        two fifty a m, 1:30 p.m.
#MONEY	        175 million Canadian Dollars, GBP 10.40
#PERCENT	    twenty pct, 18.75 %
#FACILITY	    Washington Monument, Stonehenge
#GPE	        South East Asia, Midlothian

tex='''In 965 AD, Odin, king of Asgard, wages war against the Frost Giants of Jotunheim and their leader Laufey, to prevent them from conquering the nine realms, starting with Earth. The Asgardian warriors defeat the Frost Giants and seize the source of their power, the Casket of Ancient Winters.

In the present,[N 2] Odin's son Thor prepares to ascend to the throne of Asgard, but is interrupted when Frost Giants attempt to retrieve the Casket. Against Odin's order, Thor travels to Jotunheim to confront Laufey, accompanied by his brother Loki, childhood friend Sif and the Warriors Three: Volstagg, Fandral, and Hogun. A battle ensues until Odin intervenes to save the Asgardians, destroying the fragile truce between the two races. For Thor's arrogance, Odin strips his son of his godly power and exiles him to Earth as a mortal, accompanied by his hammer Mjolnir, now protected by an enchantment that allows only the worthy to wield it.

Thor lands in New Mexico, where astrophysicist Dr. Jane Foster, her assistant Darcy Lewis, and mentor Dr. Erik Selvig, find him. The local populace finds Mjolnir, which S.H.I.E.L.D. agent Phil Coulson soon commandeers before forcibly acquiring Jane's data about the wormhole that delivered Thor to Earth. Thor, having discovered Mjolnir's nearby location, seeks to retrieve it from the facility that S.H.I.E.L.D. quickly constructed but he finds himself unable to lift it, and is captured. With Selvig's help, he is freed and resigns himself to exile on Earth as he develops a romance with Jane.

Loki discovers that he is Laufey's biological son, adopted by Odin after the war ended. A weary Odin falls into the deep "Odinsleep" to recover his strength. Loki seizes the throne in Odin's stead and offers Laufey the chance to kill Odin and retrieve the Casket. Sif and the Warriors Three, unhappy with Loki's rule, attempt to return Thor from exile, convincing Heimdall, gatekeeper of the Bifröst—the means of traveling between worlds—to allow them passage to Earth. Aware of their plan, Loki sends the Destroyer, a seemingly indestructible automaton, to pursue them and kill Thor. The warriors find Thor, but the Destroyer attacks and defeats them, prompting Thor to offer himself instead.
Struck by the Destroyer and near death, Thor's sacrifice proves him worthy to wield Mjolnir. The hammer returns to him, restoring his powers and enabling him to defeat the Destroyer. Kissing Jane goodbye and vowing to return, he and his fellow Asgardians leave to confront Loki.

In Asgard, Loki betrays and kills Laufey, revealing his true plan to use Laufey's attempt on Odin's life as an excuse to destroy Jotunheim with the Bifröst Bridge, thus proving himself worthy to his adoptive father. Thor arrives and fights Loki before destroying the Bifröst Bridge to stop Loki's plan, stranding himself in Asgard. Odin awakens and prevents the brothers from falling into the abyss created in the wake of the bridge's destruction, but Loki allows himself to fall when Odin rejects his pleas for approval. Thor makes amends with Odin, admitting he is not ready to be king; while on Earth, Jane and her team search for a way to open a portal to Asgard.

In a post-credits scene, Selvig has been taken to a S.H.I.E.L.D. facility, where Nick Fury opens a briefcase and asks him to study a mysterious cube-shaped object,[N 3] which Fury says may hold untold power. An invisible Loki prompts Selvig to agree, and he does.'''


def norm(word,pos='x'):                 	#lemmatize all nouns and no need to lemmatize proper nouns
    word = word.lower()
    if pos not in ['NNP','NNPS']:
        wnl = WordNetLemmatizer()
        word = wnl.lemmatize(word, 'n')
    return (word)     
 
 
sentList = sent_tokenize(tex)				#list of all tokenized sentences
            
sent_NounDict = defaultdict(list)        	# a dictionary key:sentence_number value:all nouns in the sentence... (nouns are normalised)

#extract only nouns from each sentence
for s in sentList:
    for w,pos in pos_tag(word_tokenize(s)):
        if pos in ['NN','NNS','NNP','NNPS']:
            sent_NounDict[sentList.index(s)].append(norm(w,pos))

wordpos_SentDict = defaultdict(list)    	# a dictionary key:(word,pos) value:all sentences it appears in...(word is normalised)

for s in sentList:
    for w,pos in pos_tag(word_tokenize(s)):
        wordpos_SentDict[(norm(w,pos),pos)].append(sentList.index(s))     


listOfTaggedSents = []      

for s in sentList:
    l = [(word,pos) for word,pos in pos_tag(word_tokenize(s))]
    listOfTaggedSents.append(l)

mostSigNoun = []                #most recently encountered significant noun
mostSigNounObject =  []         #most recently encountered significant noun which is not a person
mostSigNounPerson = []          #most recently encountered significant noun which has named entity as person

pronounsent_nounDict = defaultdict(list) 	#key:tuple(pronoun,sentence_num) val:list(list(tuple(noun,pos)))     noun not normalized

grammar =   """NP:{<DT>?<JJ>*(<NN.*>)+}    
               PR:{<PRP.*>}
            """

#grammar for tagging noun phrases and pronouns
#DT - determiners eg: The, a, an, my
#JJ - adjectives
#NN.* - any type of noun
#PRP - personal pronoun eg: He, she, I, We, they

rp = RegexpParser(grammar)
count = 0
for s in listOfTaggedSents:
    
    chunkedTree = ParentedTree.convert(rp.parse(s))         #tree of chunked parts of the sentence
                                                            #ParentedTree is used to convert tagged words to tree structure 
    neTree = ne_chunk(s)                                    #tree with named entity tags
                                                             
    #print (chunkedTree)
    #chunkedTree.draw()
    #neTree.draw()
    
    for n in chunkedTree:
        if isinstance(n,nltk.tree.Tree):            
            if n.label()=='NP':
                mostSigNoun = [w for w in n if w[1] in ['NN','NNS','NNP','NNPS']]
                for ne in neTree:                           #ne contains nouns and pos
                    if isinstance(ne,nltk.tree.Tree):
                        if ne[0] in mostSigNoun:
                            if ne.label() == 'PERSON' or ne.label() == 'ORGANIZATION' :
                                mostSigNounPerson = []  
                                mostSigNounPerson.append(ne[0])
                            else:
                                mostSigNounObject = []  
                                mostSigNounObject.append(ne[0])  
                        
            if n.label()=='PR':
                pron = n[0][0].lower()
                #print pron
                if pron in ['it','its']:    		#for objects
                    if len(mostSigNounObject)>0:        
                        pronounsent_nounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNounObject)
                    else:   						#if mostsignounobject does not exist
                        pronounsent_nounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNoun)
                else:
                    if len(mostSigNounPerson)>0:
                        pronounsent_nounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNounPerson)
                    else:  #when mostSigNounPerson is 0 then append mostSigNoun of last iteration. This else works when the sentence starts from pronoun such as he/she.
                        pronounsent_nounDict[(pron,listOfTaggedSents.index(s))].append(mostSigNoun)    
                 
                
                #adding the nouns corresponding to the pronouns to sent_NounDict and wordpos_sentdict
                for nouns in pronounsent_nounDict[(pron,listOfTaggedSents.index(s))]:   
                    for noun in nouns:  #it is a list of lists
                        sent_NounDict[listOfTaggedSents.index(s)].append(norm(noun[0],noun[1]))
                        wordpos_SentDict[(norm(noun[0],noun[1]),noun[1])].append(listOfTaggedSents.index(s))
                           
#print (sent_NounDict,'\n\n')
#print (wordpos_SentDict,'\n\n')
#print (pronounsent_nounDict,'\n\n')

for key,val in sent_NounDict.items():    			#making the values of sent_noundict a set (to remove redundant nouns)
    val = list(set(val))
    sent_NounDict[key] = val
#print (sent_NounDict)

#following code calculates the loc_of_noun_in_each_sent between two phrases
loc_of_noun_in_each_sent = defaultdict(int)			#a dict.. key:(noun or pronoun converted to noun, sent_no) value:position in the sentence from the begining
            
for s in listOfTaggedSents:
    loc = 0
    chunkedTree = ParentedTree.convert(rp.parse(s))
    for n in chunkedTree:
        if isinstance(n,nltk.tree.Tree):            
            if n.label()=='NP':
                tempNoun = [w for w in n if w[1] in ['NN','NNS','NNP','NNPS']]
                for w in tempNoun:
                    loc_of_noun_in_each_sent[(norm(*w),listOfTaggedSents.index(s))] = loc
            if n.label()=='PR':
                pronoun = n[0][0].lower()
                tempNoun = pronounsent_nounDict[(pronoun,listOfTaggedSents.index(s))]                
                for li in tempNoun:
                    for word in li:
                        loc_of_noun_in_each_sent[(norm(word[0],word[1]),listOfTaggedSents.index(s))] = loc
        loc += 1
#print(loc_of_noun_in_each_sent)

#list of all nouns in the text
listOfNouns = list(sorted(set([norm(w,pos) for s in sentList for w,pos in pos_tag(word_tokenize(s)) if pos in ['NN','NNS','NNP','NNPS']])))

#the following code assigns relation factor between two nouns
nounGraph = np.zeros((len(listOfNouns),len(listOfNouns)))


#More frequently used nouns will have more importance than rearly referenced (i.e nouns with smaller difference can be given more value)
for key,value in sent_NounDict.items():
    for v1 in value:
        for v2 in value:
            d=0
            if v2!=v1:
                d = loc_of_noun_in_each_sent[v1,key] - loc_of_noun_in_each_sent[v2,key]
                nounGraph[listOfNouns.index(v1)][listOfNouns.index(v2)] += float((100/(abs(d)+1)))
                #print(v1+' '+v2+" "+str(d))

#print(nounGraph)
         
nounPriority = defaultdict(int)             #dict to hold noun priorities... key:noun(normalized)  value:priority
sentencePriority = defaultdict(int)         #dict to hold sentence priorities...key:sentence_num   value:priority


def calcNounPriority():                     #function calculates the noun priority(sum of weights of all the edges attached to this noun in the noungraph)
    total = 0
    i=0
    for x in nounGraph:
        total = sum(x)
        nounPriority[listOfNouns[i]]=total
        i += 1


def calcSentPriority():                     #function calculates sentence priority(sum of priorities of all nouns in the sent)
    for key,value in sent_NounDict.items():
        total = 0
        for n in value:
            total += nounPriority[n]
            sentencePriority[key] = total

calcNounPriority()
calcSentPriority()

#print (sorted(nounPriority.items(),key=lambda x:x[1], reverse=True))
#print (sorted(sentencePriority.items(),key=lambda x:x[1], reverse=True))

reducingFactor = 0.8    					#to reduce the impact of nouns that were encountered before. Lower this value if more nouns(or more diversity) in summary is required
summary = []            					#list to hold the summary

for i in range(int(len(sentencePriority)/4)):       
    summary.append(max(sentencePriority.items(),key = lambda x:x[1])) 
    j = summary[-1][0]
    
    for n in sent_NounDict[j]:
        nounPriority[n] *= reducingFactor         	#reduce the priority of all nouns in the picked sentence
    
    del sent_NounDict[j]         
    del sentencePriority[j]                         #remove the picked sentence
    calcSentPriority()                              #recalculate sentence priority

print ("\n\n")
i=1
for s in sorted(summary):
    print (i,sentList[s[0]])
    i+=1