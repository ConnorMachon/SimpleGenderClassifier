from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

####### General Data Inputs ########

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
##############

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])
print(f'Using a Decision Tree as the model, the program predicts that the gender of the given input is: {prediction}.')

# Logistic Regression
clf = LogisticRegression()
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])
print(f'Using Logistric Regression as the model, the program predicts that the gender of the given input is: {prediction}.')

# K Neighbors Classifier
clf = KNeighborsClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])
print(f'Using K Neighbors Classifier as the model, the program predicts that the gender of the given input is: {prediction}.')
