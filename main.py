from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the data set
from Hard_Coded_Classifier import HardCodedClassifier

iris = datasets.load_iris()

print("Total dataset size:")
print(iris.data.size)

# Show the data (the attributes of each instance)
print("\n1. sepal length in cm\n2. sepal width in cm\n3. petal length in cm\n4. petal width in cm\n")
print(iris.data)

# Show the target values (in numeric format) of each instance
print("\n0 = Setosa\n1 = Versicolour\n2 = Virginica\n")
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)

# data is the features, target is the label

print("\n\nCommencing data training...")
print("Splitting 70% of data for training and 30% for testing...")
# test_size is the percentage held for testing, the rest is for training
data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, test_size=0.3)

print("\n\nData to be trained:")
print("%s/%s items = %s" % (data_train.size, iris.data.size, (data_train.size / iris.data.size)))
print(data_train)
print("\nTheir targets:")
print(targets_train)

print("\n\nPreparing classifier...")
classifier = GaussianNB()
print("Training classifier...")
model = classifier.fit(data_train, targets_train)

print("\nRunning test --- sending 30% data for testing and predicting targets...")
targets_predicted = model.predict(data_test)
print("\nComparing targets tested to the targets trained...")
print("Accuracy Score:")
percent = accuracy_score(targets_test, targets_predicted)
print("{:.0%}".format(percent))

print("\n\nPreparing hard-coded classifier...")

classifier = HardCodedClassifier()
print("Training classifier...")
model = classifier.fit(data_train, targets_train)
print("\nRunning test --- sending 30% data for testing and predicting targets...")
targets_predicted = model.predict(data_test)
print("\nComparing targets tested to the targets trained...")
print("Accuracy Score:")
percent = accuracy_score(targets_test, targets_predicted)
print("{:.0%}".format(percent))
