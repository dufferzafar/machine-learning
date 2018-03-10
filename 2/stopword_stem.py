from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys

#initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

# function that takes an input file and performs stemming to generate the output file
def getStemmedDocument(inputFileName, outputFileName):
	out = open(outputFileName, 'w')
	with open(inputFileName) as f:
		docs = f.readlines()
	for doc in docs:
		raw = doc.lower()
		raw = raw.replace("<br /><br />", " ")
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = [token for token in tokens if token not in en_stop]
		stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
		documentWords = ' '.join(stemmed_tokens)
		print((documentWords), file=out)
	out.close()

# creates the new stemmed documents with the suffix 'new' for both train and test files
old_file=sys.argv[1]
#with open(old_file, encoding='utf-8', errors='ignore') as f:
#	x = f.readlines()
new_file=sys.argv[2]
getStemmedDocument(old_file, new_file)
