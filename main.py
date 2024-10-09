# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # For converting text to numerical data
from sklearn.model_selection import train_test_split         # For splitting the dataset
from sklearn.naive_bayes import MultinomialNB                # The Naive Bayes classifier
from sklearn.metrics import accuracy_score                   # For evaluating the model

import requests
from io import StringIO

# Fetch the data using requests
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
response = requests.get(url)
response.raise_for_status()  # Check for HTTP errors

# Read the data into a pandas DataFrame
data = pd.read_csv(StringIO(response.text), sep='\t', header=None, names=['label', 'message'])

# Load the dataset
# We'll use a dataset of SMS messages labeled as 'ham' (not spam) or 'spam'
# url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
# data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Display the first few rows
print(data.head())

# Encode the labels ('ham' as 0 and 'spam' as 1)
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Separate the features and the target variable
X = data['message']
y = data['label_num']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Convert text data into numerical data using CountVectorizer
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)  # Learn vocabulary and transform training data
X_test_dtm = vect.transform(X_test)        # Transform testing data

# Instantiate the classifier and fit it to the training data
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# Make predictions on the test data
y_pred = nb.predict(X_test_dtm)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Test the model with a new message
new_message = ["Congratulations! You've won a $1,000 gift card. Click here to claim your prize."]
new_message_dtm = vect.transform(new_message)
prediction = nb.predict(new_message_dtm)
print('Spam' if prediction[0] else 'Not Spam')