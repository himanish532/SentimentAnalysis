import sys
import time
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer,f1_score
import string
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import sklearn
import nltk

port = nltk.stem.PorterStemmer()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Give annotations"
        sys.exit(1)

    data_dir = sys.argv[1]
    classes = ['pos', 'neg']
    cv = 3
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    All_data = []
    UnProc_Data=[]
    All_labs = []
    reader = csv.DictReader(open(data_dir,'rb'))
    i = 0
    for row in reader:
        try:
            if i < int(sys.argv[2]):
            	# Data Pre- processing
                sentence = str(row['SentimentText']).replace(string.punctuation,"") # remove punctuation
                stemmed_words=[port.stem(str(word)) for word in sentence.split(" ")] # stemming
                filtered_words = [word for word in stemmed_words if word not in stopwords.words('english')] # stop word removal
                UnProc_Data.append(row['SentimentText']) 
                All_data.append(" ".join(word for word in filtered_words))
                if int(row['Sentiment']) == 0:
                    All_labs.append('neg')
                else:
                    All_labs.append('pos')
                i+=1
        except:
            continue

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                         'C': [1, 10, 100, 1000, 10000, 100000, 1000000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    train_data, test_data, train_labels, test_labels = train_test_split(All_data, All_labs, test_size = 0.4, random_state = 0)
    un_train_data, un_test_data, un_train_labels, un_test_labels = train_test_split(UnProc_Data, All_labs, test_size = 0.4, random_state = 0)

    # Create feature vectors
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data) # Processed feature vectors
    test_vectors = vectorizer.transform(test_data)
    un_train_vectors = vectorizer.fit_transform(un_train_data) # Unprocessed feature vectors
    un_test_vectors = vectorizer.transform(un_test_data)

    # Perform classification with Optimized SVM
    classifier_rbf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cv, scoring=make_scorer(f1_score, pos_label='pos', average='weighted'),n_jobs=7)

    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Print results in a nice table
    print("Results for Optimized SVM - processed data")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print
    print sklearn.metrics.confusion_matrix(test_labels,prediction_rbf)

    t0 = time.time()
    classifier_rbf.fit(un_train_vectors, un_train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(un_test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Print results in a nice table
    print("Results for Optimized SVM - un processed data")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(un_test_labels, prediction_rbf))
    print
    print sklearn.metrics.confusion_matrix(un_test_labels,prediction_rbf)

    # Perform classification with NB
    classifier_nb = MultinomialNB()
    t0 = time.time()
    classifier_nb.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_nb = classifier_nb.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Print results in a nice table
    print("Results for Naive Bayes - processed data")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_nb))
    print
    print sklearn.metrics.confusion_matrix(test_labels,prediction_nb)

    classifier_nb = MultinomialNB()
    t0 = time.time()
    classifier_nb.fit(un_train_vectors, un_train_labels)
    t1 = time.time()
    prediction_nb = classifier_nb.predict(un_test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Print results in a nice table
    print("Results for Naive Bayes - un processed data")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(un_test_labels, prediction_nb))
    print
    print sklearn.metrics.confusion_matrix(un_test_labels,prediction_nb)



    