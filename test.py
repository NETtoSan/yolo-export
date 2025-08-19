from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
emails = [
"Win a free ticket now",
"Meeting agenda for today",
"Click here to win cash",
"Lunch with team at noon",
]

labels = [1, 0, 1, 0]

# Train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Test
test_email = ["Congratulations! You have won a discont coupon of RTX 5090 at 10 baht!"]
X_test = vectorizer.transform(test_email) # Transform the test text
prediction = model.predict(X_test) # Predict
print("Spam" if prediction[0] == 1 else "Not Spam")