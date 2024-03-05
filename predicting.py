import sys
import pandas as pd
import re
import nltk
import joblib


general_cap_classification = sys.argv[1]  # First input argument
summary = sys.argv[2]  # Second input argument

general_cap_classification = general_cap_classification.strip()
summary = summary.strip()

# if any is empty, return "low"
if general_cap_classification == '' or summary == '':
    print('low')

def cleanAndTokenizeText(txt):
    # Remove "According to the complainant,"
    txt = txt.replace('According to the complainant,', '')

    # Remove dates
    date_pattern= r'(?:\,\s)*(?:on\s)*\d+-\d+-\d+[\s]*[\,]*'
    txt = re.sub(date_pattern, '', txt, flags=re.IGNORECASE)

    # Remove time
    time_pattern= r'(at)?\s?\d+:\d+\s?(AM|PM)?\,?\s?'
    txt = re.sub(time_pattern, '', txt, flags=re.IGNORECASE)

    # Remove locations of patter of (the\s)?\d+\w+\s?District
    location_pattern= r'(the\s)?\d+\w+\s?District'
    txt = re.sub(location_pattern, '', txt, flags=re.IGNORECASE)

    # Remove "While in the confines of ,"
    txt = txt.replace("While in the confines of ,", '')

    words = nltk.tokenize.word_tokenize(txt)

    # make all lower case
    words = [word.lower() for word in words]

    #stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Remove stopwords
    words = [word for word in words if word.lower() not in stopwords]

    # Remove punctuation
    words = [word for word in words if word.isalnum()]

    # Stemming
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Remove numbers
    words = [word for word in words if not word.isdigit()]

    # remove only one-letter words
    words = [word for word in words if len(word) > 1]

    # return text of words
    return ' '.join(words)


# Create a dataframe with text as column which combies the general_cap_classification and summary
df = pd.DataFrame({'text': [general_cap_classification + '. ' + summary]})

# Clean text
df['text'] = df['text'].apply(cleanAndTokenizeText)

# CountVectorizer
count_vectorizer = joblib.load('vectorizer.pkl')

# Transform the text
X = count_vectorizer.transform(df['text'])


# Load the model
model = joblib.load('model_lr_cv.pkl')


# Predict the label
label = model.predict(X )[0]

# Return the label
print(label)