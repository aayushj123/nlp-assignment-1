
# Long words (longer than 15 characters)
long_words = re.findall(r'\w{15,}', corpus)

# Words with hyphen
hyphenated_words = re.findall(r'\w+-\w+', corpus)

# Words ending with common morphological suffixes, also indicating verbs
morphological_suffixes_words = re.findall(rf'\b\w+ed|\w+ing|\w+ly\b', corpus)

# proper nouns
proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', corpus)

print("\nLong words (longer than 15 characters): ", long_words)
print("\nHyphenated words: ", hyphenated_words)
print("\nCommon morphological suffixes indicating verb: ", morphological_suffixes_words)
print("\nProper nouns: ", proper_nouns)

# Clean the corpus
# Remove punctuations
corpus = re.sub(r'[^\w\s]', '', corpus)

# Remove numbers
corpus = re.sub(r'\d+', '', corpus)

# Remove all the extra spaces
corpus = re.sub(r'\s+', ' ', corpus)

# change all words to lower case
corpus = corpus.lower()

# Tokenize the corpus and mention the counts of top 10 words excluding stop words
tokens = nltk.word_tokenize(corpus)
tokens = [token for token in tokens if token not in stop_words]
fdist = nltk.FreqDist(tokens)
top_10 = fdist.most_common(10)

print("\nTop 10 Words: ", top_10)
