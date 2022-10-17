
# Creating corpus variable for 1st question
corpus_q1 = corpus

# Interesting regex patterns
# Long words (longer than 15 characters)
long_words = re.findall(r'\w{15,}', corpus_q1)

# Words with hyphen
hyphenated_words = re.findall(r'\w+-\w+', corpus_q1)

# Words ending with common morphological suffixes, also indicating verbs
morphological_suffixes_words = re.findall(rf'\b\w+ed|\w+ing|\w+ly\b', corpus_q1)

# Patters that indicate proper nouns
proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', corpus_q1)

print("\nLong words (longer than 15 characters): ", long_words)
print("\nHyphenated words: ", hyphenated_words)
print("\nCommon morphological suffixes indicating verb: ", morphological_suffixes_words)
print("\nProper nouns: ", proper_nouns)

# Clean the corpus
# Remove all the punctuations
corpus_q1 = re.sub(r'[^\w\s]', '', corpus_q1)

# Remove the numbers
corpus_q1 = re.sub(r'\d+', '', corpus_q1)

# Remove all the extra spaces
corpus_q1 = re.sub(r'\s+', ' ', corpus_q1)

# Convert all the words to lower case
corpus_q1 = corpus_q1.lower()

# Tokenize the corpus and mention the counts of top 10 words excluding stop words
tokens = nltk.word_tokenize(corpus_q1)
tokens = [token for token in tokens if token not in stop_words]
fdist = nltk.FreqDist(tokens)
top_10 = fdist.most_common(10)

print("\nTop 10 Words: ", top_10)
