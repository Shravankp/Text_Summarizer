# Text_Summarizer
A text summarizer that extracts important sentences based on noun-pronoun distance relationship. Uses pos tagging and tree structure for lexical analysis. 

Text summarization can broadly be divided into two categories — Extractive Summarization and Abstractive Summarization. 
This project is based on extractive summarization where in important sentences are extracted to form a summary.(Does not generate an entire new summary)

Importance of a sentence is measured by lexical analysis and by measuring noun-pronoun relationship and thier distances. 
* Preprocess the text i.e to remove stopwords, stemming and lemmatizing.
* Identify which words are nouns and pronouns.
* Forming an lexical tree to distinguish person, object, group and so on.
* Finding relation between nouns and pronouns to Assign nouns to each of the pronouns to which it referred to.
* whichever noun-pronoun pair has less distance and occur frequently in a sentence is considered as an important sentence.
  (The same can be implemented for noun-verb, noun-preposition relationship and so on.)

Nlp involves iterating the text many times. To avoid it as much as possible it is programmed using many dictionaries.
Some of the objects (dictionaries, lists) used are (just for the reference): <br/> 
Dictionary Name         | Key                 | Values
------------------------|---------------------|------------------------------
sent_NounDict           | Sentence number     | Nouns
wordpos_SentDict        | (word, pos)         | Sentence number
pronounSent_nounDict    | (pronoun, sentence no)     | list(list((noun, pos)))
nounPriority            | noun                | priority 
sentencePriority        | Sentence number     | priority 

mostSigNoun <br/> List of most significant nouns <br/> 
mostSigNounObject - List of most significant nouns which are objects <br/> 
mostSigNounPerson - List of most significant nouns Person <br/> 
sentList - List of sentences <br/> 
listOfNouns - List of nouns <br/> 

