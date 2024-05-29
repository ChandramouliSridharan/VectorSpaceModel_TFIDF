# VectorSpaceModel_CosineSimilarity
This Code will read a corpus and produce TF-IDF vectors for documents in the corpus. Then,given a query string, our code will return the query answer--the document with the highest cosine similarity score for the query.

We need to use the corpus of 30 Inaugural addresses of different US presidents. This corpus is processed and it is available in the **US_Inagural_Addresses** folder, which includes 30 .txt files.

**VectorSpaceModel.py** is a single python file containing the Cosine Similarity for Query document.

**Tasks Performed:**
  1. **Reading** the 30 .txt files, each of which has the transcript of inaugural addresses by different US presidents.
  2. Converting the text to **lower case** before doing anything with the text. For a query, also converting to lower case.
  3. **Tokenize** the content of each file using **RegexpTokenizer**.
  4. **Removing stopwords** on the obtained tokens using NLTK Stopwords Corpus data.
  5. Performing **stemming** on the obtained tokens using NLTK **Porter stemmer**.
  6. Using the tokens, we are **computing the TF-IDF** vector for each document.
  7. **getidf(token): **return the inverse document frequency of a token. If the token doesn't exist in the corpus, it returns -1.
  8. **getweight(filename,token)**: return the normalized TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, returning 0.
  9. **query(qstring)**: return a tuple in the form of (filename of the document, score), where the document is the query answer with respect to the weighting scheme.
