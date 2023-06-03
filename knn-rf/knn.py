import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Load data from file
with open('data_train.json', 'r') as f:
    train_data = json.load(f)
with open('data_dev.json', 'r') as f:
    dev_data = json.load(f)
with open('data_test.json', 'r') as f:
    test_data = json.load(f)

train_data = train_data + dev_data

# Get input data sets
train_premises = [example['premise'] for example in train_data]
train_hypotheses = [example['hypothesis'] for example in train_data]
train_labels = [example['gold_label'] for example in train_data]

test_premises = [example['premise'] for example in test_data]
test_hypotheses = [example['hypothesis'] for example in test_data]
test_labels = [example['gold_label'] for example in test_data]

# Vectorize premise and hypothesis texts using a pre-trained word embedding model
vectorizer = CountVectorizer()

vectorizer.fit(train_premises)
vectorizer.fit(train_hypotheses)

vectorizer.fit(test_premises)
vectorizer.fit(test_hypotheses)

train_X = np.concatenate(
    [vectorizer.transform(train_premises).toarray(),
     vectorizer.transform(train_hypotheses).toarray()], axis=1)
test_X = np.concatenate(
    [vectorizer.transform(test_premises).toarray(),
     vectorizer.transform(test_hypotheses).toarray()], axis=1)

train_y = np.array(train_labels)

# Create a KNN classifier with 
k=5
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier to the training data
knn.fit(train_X, train_y)

# Make predictions on the test set
pred_y = knn.predict(test_X)

# Evaluate the classifier on the test data
accuracy = knn.score(test_X, test_labels)
print("Accuracy:", accuracy)

# Some needed metrics
pred_0 = 0
pred_1 = 0
success = 0
fail = 0
for i, actual in enumerate(test_labels):
	prediction = pred_y[i]
	
	if prediction == 0:
		pred_0 += 1
	else:
		pred_1 += 1

	if actual == prediction:
		success += 1
	else:
		fail += 1

print("predited 0s:", pred_0, "rate:", pred_0/len(test_labels))
print("predited 1s:", pred_1, "rate:", pred_1/len(test_labels))

print()
print("success rate:", success/len(test_labels))
"""
"""