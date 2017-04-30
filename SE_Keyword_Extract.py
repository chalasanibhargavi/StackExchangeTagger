# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
import re
from time import time
import operator
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from scipy.sparse import coo_matrix

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import metrics
from bs4 import BeautifulSoup, Tag



# used for cleaning the code part in the body of post
def clean_html(text):
    soup = BeautifulSoup(text, "lxml")
    for tag in soup.find_all('code'):
        tag.decompose()
    return soup.get_text()



# split the tag column in to text
# ex: <java><jquery><php> ----> java, jquery, php
def tag_split(text):
    return re.findall(r"\<(.*?)\>", text)


# to get the tags in x;y;z format
def append_tag(tag_list):
    return ';'.join(tag_list)


# tokenize the given text
def my_tokenizer(s):
    return s.split(";")


def read_data(input):

    df_posts = pd.read_csv(input)
    df_posts.head()

    #shuffle the rows (as data from different domains is selected)
    df_posts = df_posts.sample(frac=1).reset_index(drop=True)

    #drop rows where number of tags is 0
    df_posts.drop(df_posts[df_posts['Tags'].isnull()].index, inplace=True)
    print("Number of rows with zero tags:")
    print(len(df_posts[df_posts['Tags'].isnull()]))

    # clean the 'code' part in the body
    df_posts['Body'] = df_posts['Body'].apply(clean_html)

    # remove new line character in the body and combine the title of the post with body
    df_posts['Body'] = df_posts['Body'].str.replace("\n", "")
    df_posts['Full_Text'] = df_posts['Title'] + df_posts['Body']

    # split the tags
    df_posts['split_tags'] = df_posts['Tags'].apply(tag_split)

    df_posts_merged = df_posts[['Id', 'Full_Text', 'split_tags']]
    df_posts_merged['Tag_text'] = df_posts_merged['split_tags'].apply(append_tag)

    return df_posts_merged



"""
    Function
    --------
    makeX(Text, X_min_df)

    Converts Text to vectorized representation X

"""


def makeX(Text, X_min_df=.0001):
    X_vectorizer = TfidfVectorizer(min_df=X_min_df, sublinear_tf=True, max_df=0.9, stop_words='english')
    X = X_vectorizer.fit_transform(Text)
    return coo_matrix(X), X_vectorizer


"""
    Function
    --------
    makeY(Tags, Y_min_df)

    Converts Tags to vectorized representation Y


"""


def makeY(Tags, Y_min_df=.0001):
    Y_vectorizer = CountVectorizer(tokenizer=my_tokenizer, min_df=Y_min_df, binary=True)
    Y = Y_vectorizer.fit_transform(Tags)
    return Y, Y_vectorizer


# In[29]:

"""
    Takes input dfmatrix and returns the matrix of predicted tags.

"""


def df_to_preds(dfmatrix, k=5):
    predsmatrix = np.zeros(dfmatrix.shape)
    for i in range(0, dfmatrix.shape[0]):
        dfs = list(dfmatrix[i])
        if (np.sum([int(x > 0.0) for x in dfs]) <= k):
            predsmatrix[i, :] = [int(x > 0.0) for x in dfs]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(dfs), key=operator.itemgetter(1), reverse=True)[0:k]]
            listofzeros = [0] * len(dfs)
            for j in range(0, len(dfs)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i, :] = listofzeros
    return predsmatrix


"""

    Takes input probsmatrix and returns the matrix of predicted tags.


"""


def probs_to_preds(probsmatrix, k=5):
    predsmatrix = np.zeros(probsmatrix.shape)
    for i in range(0, probsmatrix.shape[0]):
        probas = list(probsmatrix[i])
        if (np.sum([int(x > 0.01) for x in probas]) <= k):
            predsmatrix[i, :] = [int(x > 0.01) for x in probas]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(probas), key=operator.itemgetter(1), reverse=True)[0:k]]
            listofzeros = [0] * len(probas)
            for j in range(0, len(probas)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i, :] = listofzeros
    return predsmatrix



"""
    Function
    --------
    opt_params(clf_current, params)

    Determines the set of parameters for clf_current that maximize cross-validated F1 score using sklearn's GridSearchCV

"""


def opt_params(clf_current, params):
    model_to_set = OneVsRestClassifier(clf_current)
    grid_search = GridSearchCV(model_to_set, param_grid=params, scoring="f1_weighted")

    print("Performing grid search on " + str(clf_current).split('(')[0])
    print("parameters:")
    print(params)
    grid_search.fit(X_train, Y_train.toarray())
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    gs = grid_search.grid_scores_
    ret = [(i[0], i[1]) for i in gs]
    return best_parameters, ret


# In[91]:

"""
    Function
    --------
    benchmark(clf_current)

    Takes the classifier passed and determines how well it performs on the test set.


"""


def benchmark(clf_current):
    print('_' * 80)
    print("Test performance for: ")
    clf_descr = str(clf_current).split('(')[0]
    print(clf_descr)
    t0 = time()
    classif = OneVsRestClassifier(clf_current)
    classif.fit(X_train, Y_train.toarray())
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    if hasattr(clf_current, "decision_function"):
        dfmatrix = classif.decision_function(X_test)
        score = metrics.f1_score(Y_test.toarray(), df_to_preds(dfmatrix, k=5), average="macro")
    else:
        probsmatrix = classif.predict_proba(X_test)
        score = metrics.f1_score(Y_test.toarray(), probs_to_preds(probsmatrix, k=5), average="macro")

    test_time = time() - t0

    print("f1-score:   %0.7f" % score)
    print("test time:  %0.3fs" % test_time)

    print('_' * 80)
    return clf_descr, score, train_time, test_time


# Initialize the parameters for different classifiers
def init_params():
    classlist = [
        (LinearSVC(), {'estimator__penalty': ['l1', 'l2'], 'estimator__loss': ['l2'], 'estimator__dual': [False],
                       'estimator__tol': [1e-2, 1e-3]}),
        (SVC(), {'estimator__kernel': ['rbf', 'poly', 'sigmoid']}),
        (MultinomialNB(), {"estimator__alpha": [.01, .1], "estimator__fit_prior": [True, False]}),
        (BernoulliNB(), {"estimator__alpha": [.01, .1], "estimator__fit_prior": [True, False]})
    ]

    return classlist


# optimize the params for the list of initialized classifiers and then execute it on the test set
def run_opt_test(classlist, optimize_params = 0):
    results = []
    classifier_pickle = []

    if optimize_params == 1:
        for classifier, params_to_optimize in classlist:
            best_params, gs = opt_params(classifier, params_to_optimize)
            classifier_pickle.append(best_params['estimator'])

        pickle.dump(classifier_pickle, open("classifierpickle" + ".p", "wb"))

    else:
        classifier_pickle = pickle.load(open("classifierpickle" + ".p", "rb"))
        for best_paramet in classifier_pickle:
            results.append(benchmark(best_paramet))

    return results



# output results to a csv file
def output_csv(results):
    df_res = pd.DataFrame(results, columns=['Classifier', 'F1-Score', 'Train-Time', 'Test-Time'])
    df_res.to_csv('rtest_results.csv')


"""
    Function
    --------
    plot_results(current_results)

    Takes results of benchmarking function and plot

    Parameters
    ----------

    current_results : list
        Output of the benchmark function

    title : str
        The title for the plot

    Returns
    -------
    None

"""


# make some plots
def plot_results(current_results, title="Score"):
    indices = np.arange(len(current_results))

    results2 = [[x[i] for x in current_results] for i in range(4)]

    clf_names, score, training_time, test_time = results2
    max_train_time = np.max(training_time)
    training_time = np.array(training_time) / max_train_time
    test_time = np.array(test_time) / max_train_time

    plt.figure(1, figsize=(14, 5))
    plt.title(title)
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.xlabel('Mean F1 Score (time values are indexed so max training time = 1.0)')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.2, i, c)

    plt.show()



import warnings

warnings.filterwarnings("ignore")

df_posts_merged = read_data("SOPosts_combined.csv")

Text = df_posts_merged['Full_Text'].tolist()[:1000]
Tags = df_posts_merged['Tag_text'].tolist()[:1000]

Y, vectorizer2 = makeY(Tags, Y_min_df=int(1))
X, vectorizer1 = makeX(Text, X_min_df=int(10))
X_current = X
X_train, X_test, Y_train, Y_test = train_test_split(X_current, Y)

print("Classifying data of size Train: " + str(X_train.shape[0]) + " Test: " + str(X_test.shape[1]))

classlist = init_params()

results = run_opt_test(classlist, 0) # 1 - to generate optimal parameters, 0 - to use generated parameters

if results:
    plot_results(results, title="Classifier F1 Results on Tf-idf vector")
    output_csv(results)

else:
    print("Successfully generated optimal parameters")
    print("Now call run_opt_test(classlist, 0) to test")

