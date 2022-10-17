
corpus_q3 = ""

# Preprocessing the corpus
for line in corpus:
    # Remove newlines
    line_nl_removed = line.replace("\n", " ")
    corpus_q3 += line_nl_removed

# Removing spl characters
corpus_q3 = "".join([char for char in corpus_q3 if char not in string.punctuation])

# Sentence tokenize the corpus
sentences = nltk.sent_tokenize(corpus_q3)
print("Number of sentences: ", len(sentences))

# Word tokenize the corpus
words = nltk.word_tokenize(corpus_q3)
print("Number of tokens: ", len(words)) 

# Average number of tokens per sentence
average_tokens = round(len(words)/len(sentences))
print("Average number of tokens per sentence: ",
average_tokens) 

# Number of unique tokens
unique_tokens = set(words)
print("Number of unique tokens: ", len(unique_tokens)) 

# Create language models
unigram = []
bigram = []
trigram = []
tokenized_text = []

for sentence in sentences:
    sentence = sentence.lower()
    sequence = word_tokenize(sentence) 
    for word in sequence:
        if (word =='.'):
            sequence.remove(word) 
        else:
            unigram.append(word)
    tokenized_text.append(sequence)
    # Create bigrams
    bigram.extend(list(ngrams(sequence, 2)))
    # Create trigrams
    trigram.extend(list(ngrams(sequence, 3)))

# Remove stop words from unigram, bigram and trigram
unigram_stopwords_removed = [word for word in unigram if word not in stop_words]
bigram_stopwords_removed = list(ngrams(unigram_stopwords_removed, 2))
trigram_stopwords_removed = list(ngrams(unigram_stopwords_removed, 3))

# Frequency distribution of unigrams, bigrams and trigrams
fdist_unigram = nltk.FreqDist(unigram_stopwords_removed)
fdist_bigram = nltk.FreqDist(bigram_stopwords_removed)
fdist_trigram = nltk.FreqDist(trigram_stopwords_removed)

# Count of unique unigrams, bigrams and trigrams
unique_unigrams = set(unigram_stopwords_removed)
unique_bigrams = set(bigram_stopwords_removed)
unique_trigrams = set(trigram_stopwords_removed)

print("\nNumber of unique unigrams: ", len(unique_unigrams))
print("Number of unique bigrams: ", len(unique_bigrams))
print("Number of unique trigrams: ", len(unique_trigrams))

# Print top 10 unigrams, bigrams and trigrams
print("\nMost common n-grams with stopword removal: ")

print("\nTop 10 Unigrams: ", fdist_unigram.most_common(10))
print("\nTop 10 Bigrams: ", fdist_bigram.most_common(10))
print("\nTop 10 Trigrams: ", fdist_trigram.most_common(10))

# Applying Add One smoothing
ngrams_all = {1:[], 2:[], 3:[]}

for i in range(3):
    for each in tokenized_text:
        for j in ngrams(each, i + 1):
            ngrams_all[i + 1].append(j)

ngrams_vocabulary = {1:set([]), 2:set([]), 3:set([])}

for i in range(3):
    for gram in ngrams_all[i + 1]:
        if gram not in ngrams_vocabulary[i + 1]:
            ngrams_vocabulary[i + 1].add(gram)

total_ngrams = {1: -1, 2: -1, 3: -1}
total_vocabulary = {1:-1, 2:-1, 3:-1}

for i in range(3):
    total_ngrams[i + 1] = len(ngrams_all[i + 1])
    total_vocabulary[i + 1] = len(ngrams_vocabulary[i + 1])                       
    
ngrams_probabilities = {1:[], 2:[], 3:[]}

for i in range(3):
    for ngram in ngrams_vocabulary[i + 1]:
        temp_list = [ngram]
        temp_list.append(ngrams_all[i + 1].count(ngram))
        ngrams_probabilities[i + 1].append(temp_list)
    
for i in range(3):
    for ngram in ngrams_probabilities[i + 1]:
        ngram[-1] = (ngram[-1] + 1)/(total_ngrams[i + 1] + total_vocabulary[i + 1])

# Print top 10 unigram, bigram, trigram after add one smoothing
print("\nMost common n-grams without stopword removal and with add one smoothing: ")

for i in range(3):
    ngrams_probabilities[i + 1] = sorted(ngrams_probabilities[i + 1], key = lambda x:x[1], reverse = True)
    
print ("\nMost common unigrams: ", str(ngrams_probabilities[1][:10]))
print ("\nMost common bigrams: ", str(ngrams_probabilities[2][:10]))
print ("\nMost common trigrams: ", str(ngrams_probabilities[3][:10]))

# Predict next word
to_predict = "to be"

# Tokenize 
tokens_of_to_predict = nltk.word_tokenize(to_predict)

ngrams_of_to_predict = {1:[], 2:[]}

for i in range(2):
    ngrams_of_to_predict[i + 1] = list(ngrams(tokens_of_to_predict, i + 1))[-1]

for i in range(3):
    ngrams_probabilities[i + 1] = sorted(ngrams_probabilities[i + 1], key = lambda x:x[1], reverse = True)
    
prediction = {1:[], 2:[], 3:[]}

for i in range(2):
    count = 0
    for each in ngrams_probabilities[i + 2]:
        # Find predictions based on highest probability of ngrams
        if each[0][:-1] == ngrams_of_to_predict[i + 1]:
            count += 1
            prediction[i + 1].append(each[0][-1])
            if count == 5:
                break
    if count < 5:
        while(count != 5):
            prediction[i + 1].append("No Prediction")
            count += 1

print(f'\nTop 5 predictions for "{to_predict}" are: ')
print("\nBigram model predictions: ", prediction[1])
print("\nTrigram model predictions: ", prediction[2])
