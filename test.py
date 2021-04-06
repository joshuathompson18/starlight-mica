
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print("Libraries finished loading")

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#describe just provides basic stats
#print(dataset.describe())
#.groupby() categorizes things by class
#the .size() helps count the count of indices
#print(dataset.groupby('class'))

# box and whisker plots
#.PLOT is from matplot
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()
#
## histograms
#dataset.hist()
#pyplot.show()
#
## scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()

# Split-out validation dataset
#convert dataframe to numpy array
#basically, converting lists to array
array = dataset.values
#split the array into x and y variables
X = array[:,0:4]
y = array[:,4]

#split into subsets
#I think we are going to use the dataset to predict the
#first arg is array, second arg is test size
#it knows how to split into x and y?
#This is purely just for randomizing data, so we don't get biased shit
#I'm assuming x is input and y is output
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#x_train is the input
#y_train is expected

# Spot Check Algorithms
#creating a list of models
#each entry is a string, and class
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model with kfold
results = []
names = []
for name, model in models:
    #establish kfold criteria
    #n_splits is how many chunks of data we split x_Train into
    #random_state is the seed for randomizing our x_Train data
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    #run and return cross_val
    #kfold splits up our dataset, shuffles data set, generaters a model based on a subset(fold) of the training data, then predicts the remaining data using the trained model parameters
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
    #printing both the mean score of all the kfolds and the standard deviation. High STD means that there's a case in which the results are not very consistent. M
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))




# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#define our model and its configurations
model = SVC(gamma='auto')
#generate the model parametetrs
model.fit(X_train, Y_train)
#provide the validation/untested inputs
predictions = model.predict(X_validation)

#compare accuracy between expect data and prediction
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))