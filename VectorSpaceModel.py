import os
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

# Give your file path.
corpusroot = './US_Inaugural_Addresses'
docContent={}
term = {}
N = 30

# Performs Tokenization and Stemming for the input parameter.
def tokenizeAndStemming(docValue):
    docValue = docValue.lower()
    # Tokenize the document using Regex expression
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokensList = tokenizer.tokenize(docValue)                 
        
    # Stopwords List
    stopwordsList= stopwords.words('english')

    # Using Porter Stemmer algorithm
    stemmer = PorterStemmer()
    stemmedTokens = [stemmer.stem(token) for token in tokensList if token not in stopwordsList]

    return stemmedTokens

# Traverse all 30 files and generate the tokens that are stemmed.
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1') or filename.startswith('2') or filename.startswith('3'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()        
        docContent[filename] = tokenizeAndStemming(doc)
        count = Counter(docContent[filename])
        for key, value in count.items():
            if key not in term:
                term.setdefault(key, 1)
            else:
                term[key] += 1

# Calculating the Weighted TF for all terms in all documents.
def getWtf(doc):
    result = {}
    for filename, content in doc.items():
        tf = Counter(content)
        for key, value in tf.items():
            if value != 0:
                tf[key] = 1 + math.log10(tf[key])
        result.setdefault(filename, tf)
    return result

# Calculating the IDF for all terms based on the Document frequency score.
def getDocumentidf(term):
    temp = {}
    for key, value in term.items():
        temp.setdefault(key, math.log10(N / value))
    return temp

# Calculating the Magnitude for all files based on the TF IDF scores.
def getMagnitude(documentTF, documentIDF):
    result = {}
    for filename, tf in documentTF.items():
        sum = 0
        for key, value in tf.items():
            sum += (value*documentIDF[key]) ** 2
        result.setdefault(filename, sum ** 0.5)

    return result

documentTF = getWtf(docContent)
documentIDF = getDocumentidf(term)
documentMagnitude = getMagnitude(documentTF, documentIDF)

# Returns the IDF Score for that particular word.
def getidf(word):
    word = tokenizeAndStemming(word)
    if(len(word) == 0):
        return -1
    if word[0] not in documentIDF:
        return -1
    return documentIDF[word[0]]
   
# Returns the Normalized TF-IDF score for the given word from the mentioned file name.
def getweight(userfilename, word): 
    if(userfilename in documentTF):
        tfList = documentTF[userfilename]
        stemmedword = tokenizeAndStemming(word)
        if(len(stemmedword) ==0):
            return 0        
        if(stemmedword[0] in tfList):
            score = tfList[stemmedword[0]] * documentIDF [stemmedword[0]]
            normalizedscore= score / documentMagnitude [userfilename]
            return normalizedscore
    return 0

# Returns the maximum Cosine Similarity Score and its corresponding document file name for the given query. 
def query(word):
    stemmedQueryList = tokenizeAndStemming(word)
    queryOccurence=Counter(stemmedQueryList)
    queryScore= dict()
    maxSimilarity =dict ()
    for term , counts in queryOccurence.items():
        if counts == 0 :
            queryScore[term] = 0
        else:
            queryScore[term] = 1 + (math.log10(counts)) 

    squares = sum(x**2 for x in queryScore.values())
    queryMagnitude = math.sqrt(squares)

    for filename in docContent.keys():
        tfList= documentTF[filename]
        cosinedotproduct = sum(queryScore[term] * tfList[term] * documentIDF[term] for term in stemmedQueryList if term in tfList)
        docmagnitude = documentMagnitude[filename]
        cosinesimilarityscore = cosinedotproduct / (docmagnitude * queryMagnitude)
        maxSimilarity[filename] = cosinesimilarityscore

    maxKey = max(maxSimilarity, key=lambda k: maxSimilarity[k])
    maxValue = maxSimilarity[maxKey]
    
    return maxKey , maxValue

print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))  
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jeffe','press'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))
print("------------")
print("Special Cases , Incorrect input")
print("%.12f" % getidf('  '))
print("%.12f" % getweight('05_jefferson_1805.txt', 'AT&T'))