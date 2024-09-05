import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn. tree import DecisionTreeClassifier

class BaggingClassifier:
    def __init__ (self, base_classifier, n_estimators):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled = X[indices]
            y_sampled = y[indices]

            # Create a new base classifier and train it on the sampled data
            classifier = self.base_classifier.__class__()
            classifier.fit(X_sampled, y_sampled)
            # Store the trained classifier in the list of classifiers
            self. classifiers . append (classifier)
        return self. classifiers

    def predict(self, X):
        # Make predictions using all the base classifiers
        predictions = [classifier.predict(X) for classifier in self.classifiers]
        # Aggregate predictions using majority voting
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0,arr=predictions)
        return majority_votes
    
digit = load_digits()
X, y = digit.data, digit.target

# Split the data into training and test ing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Create the base classifier
dc = DecisionTreeClassifier()
model = BaggingClassifier(base_classifier = dc, n_estimators = 10)
classifiers = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

for i, clf in enumerate(classifiers):
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy: "+str(i+1), ':', accuracy)