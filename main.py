import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd

# Read the dataset into a DataFrame
file_path = 'Emotions.txt'
column_names = ['sentence', 'emotion']
df = pd.read_csv(file_path, sep=';', names=column_names)

# Remove duplicates
df = df.drop_duplicates()

# Convert to lowercase
df['sentence'] = df['sentence'].str.lower()

# Tokenization
df['tokens'] = df['sentence'].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['emotion'], test_size=0.2, random_state=42)

# Vectorizing the data
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize SVM classifier
svm_classifier = SVC()

# Train the model
svm_classifier.fit(X_train_tfidf, y_train)

# Get user input
user_input = input("Enter a sentence: ")

# Preprocess the user input
user_input = user_input.lower()
user_tokens = word_tokenize(user_input)
user_tokens = [word for word in user_tokens if word not in stop_words]
user_tokens = [lemmatizer.lemmatize(word) for word in user_tokens]

# Vectorize the user input
user_input_tfidf = tfidf_vectorizer.transform([user_input])

# Make predictions on the user input
user_prediction = svm_classifier.predict(user_input_tfidf)

# Display the predicted emotion
print("Predicted Emotion:", user_prediction[0])
