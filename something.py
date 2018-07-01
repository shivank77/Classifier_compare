from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB

clf1 = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier()
clf3 = SVC()
clf4 = LinearSVC()
clf5 = NuSVC()
clf6 = GaussianNB()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)
clf5 = clf5.fit(X, Y)
clf6 = clf6.fit(X, Y)


prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])
prediction4 = clf4.predict([[190, 70, 43]])
prediction5 = clf5.predict([[190, 70, 43]])
prediction6 = clf6.predict([[190, 70, 43]])
# CHALLENGE compare their reusults and print the best one!

print(prediction1)
print(prediction2)
print(prediction3)
print(prediction4)
print(prediction5)
print(prediction6)