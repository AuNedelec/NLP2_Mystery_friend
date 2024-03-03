# # Project 2 : Mystery Friend

# ## Feature Vectorizing

# importing scikit-learn in jupyter notebook
get_ipython().system('pip install scikit-learn')

# importing sklearn modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# creating a CountVectorizer object for converting text documents into a matrix of token counts
bow_vectorizer = CountVectorizer()

# importing fictive texts (provided by CodeCademy) to use :
from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs

friends_docs = goldman_docs + henson_docs + wu_docs

# training and vectorizing texts :
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

# this text will be compared with the fictive texts previously imported
mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""
# using the CountVectorizer object (bow_vectorizer) to transform the text of the mystery postcard 
# into a numerical representation of its token counts (mystery_vector).
mystery_vector = bow_vectorizer.transform([mystery_postcard])


# ## Classification

# printing out a random document from each friend in order to have a look at their writing style:
print(goldman_docs[45])
print(henson_docs[79])
print(wu_docs[123])

# implementing a Naive Bayes classifier using MultinomialNB :
friends_classifier = MultinomialNB()

# creating labels for the dataset
friends_labels = ["Emma"] * 154 + ["Matthew"] * 141 + ["Tingfang"] * 166

# training the classifier:
friends_classifier.fit(friends_vectors, friends_labels)

# use the trained classifier (friends_classifier) to predict the probability for each labels for the mystery_vector
predictions = friends_classifier.predict_proba(mystery_vector)
print(predictions)
# we can see here that there are way more probabilities that the postcard was written by Matthew.

# use the trained classifier (friends_classifier) to predict the labels for the mystery_vector
predictions = friends_classifier.predict(mystery_vector)


# ## Mystery Revealed!

mystery_friend = predictions[0] if predictions[0] else "someone else"
print("The postcard was from {}!".format(mystery_friend))

