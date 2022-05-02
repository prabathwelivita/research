import nltk, re, string, random

# Downloading words data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

# Importing required libraries and dependencies
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import request
import json

app = Flask(__name__)

# print(tweet_tokens[0])

# print(pos_tag(tweet_tokens[0]))

with open('positive.json') as f:
    positiveData = json.load(f)

with open('negative.json') as f:
    negativeData = json.load(f)


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    # TAG words with NLTK POS tagger : https://www.nltk.org/book/ch05.html
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

# print(tweet_tokens[0])
# print(lemmatize_sentence(tweet_tokens[0]))

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    # Removing urls and other unnecessary words (noise)
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        # Word net lemmatizer : https://www.programcreek.com/python/example/81649/nltk.WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        # If the lemmatized tokens are not punctuation and they are not stop words -> add those tokens to the end of cleaned tokens
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Starting NLP
stop_words = stopwords.words('english')


positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []


positive_cleaned_tokens_list = positiveData

negative_cleaned_tokens_list = negativeData

# Turn 2D array of tokens in to single 1D array
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# all_pos_words = get_all_words(positive_cleaned_tokens_list)
#
# freq_dist_pos = FreqDist(all_pos_words)
# # print(freq_dist_pos.most_common(10))

def get_words_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_words_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_words_for_model(negative_cleaned_tokens_list)


positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

# Create a single data set with both negative and positive data sets prepared
dataset = positive_dataset + negative_dataset

# Shuffle the data set to mix positive and negative data to be split to train data and test data
# random.shuffle(dataset)

# Split whole data set into train and test data
train_data = dataset[:4000]
test_data = dataset[:400]

# train the NaiveBayesClassifier model with train_data
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))


custom_text = "his product is good"

custom_tokens = remove_noise(word_tokenize(custom_text))

# Test print
print(classifier.classify(dict([token, True] for token in custom_tokens)))

# Flask API to be used in backend
@app.route("/NlpAspect")
def hello():
    # text as a http get request parameter
    custom_tweet = request.args.get('text')
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    return classifier.classify(dict([token, True] for token in custom_tokens))


custom_text = "  this product is good"

custom_tokens = remove_noise(word_tokenize(custom_text))

# Test print
print(classifier.classify(dict([token, True] for token in custom_tokens)))
if __name__ == '__main__':
    # Set the API port to 8083 and enable debug to view clear errors
    app.run(debug=True, port=8084)

