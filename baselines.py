import argparse
import os
import pickle
import sys
import time

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier  # logistic regression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB  # NB
from sklearn.neighbors import KNeighborsClassifier  # k-NN
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC  # linear SVM
from sklearn.tree import DecisionTreeClassifier  # DT


def pkl_write(data, filename='data.pickle'):
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pkl_read(filename='data.pickle'):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data


def build_cls(ml_cls="NB", tfidf=False, use_hash=False, scaler=False):
    print("- Construct the baseline...")
    start = time.time()
    if ml_cls == "kNN":
        KNeighborsClassifier(n_neighbors=5)
    elif ml_cls == "LR":
        # Logistic Regression
        classifier = SGDClassifier(verbose=5, loss='log', max_iter=100)
    elif ml_cls == "DT":
        classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    elif ml_cls == "SVM":
        classifier = LinearSVC(verbose=5)
    elif ml_cls == "MLP":
        classifier = MLPClassifier(random_state=1, max_iter=100)
    elif ml_cls == "AB":
        classifier = AdaBoostClassifier()
    elif ml_cls == "GB":
        classifier = GradientBoostingClassifier(verbose=5)
    elif ml_cls == "RF":
        classifier = RandomForestClassifier(n_estimators=100, verbose=5)
    else:
        # DEFAULT: NB
        classifier = MultinomialNB()
    settings = []
    # if use_hash:
    #     settings = [('vectorizer', HashingVectorizer())]
    # elif tfidf:
    #     settings = [('vectorizer', TfidfVectorizer())]
    # else:
    #     # DEFAULT: BOW counting
    #     settings = [('vectorizer', CountVectorizer())]

    # scaler cannot use with NB (MultinomialNB)
    if scaler and ml_cls != "NB":
        settings += [('scaler', StandardScaler())]

    settings += [('classifier', classifier)]
    model = Pipeline(settings)

    for key in model.get_params():
        print(key)
    print(model.get_params())
    # parameters = {  # 'vectorizer__analyzer': ['word'],
    #     # 'vectorizer__analyzer': ['char', 'word'],
    #     'vectorizer__ngram_range': [(1, 5)],
    #     'vectorizer__min_df': [3, 5, 7],
    #     'vectorizer__binary': (True, False)
    # }

    # parameters = {'vectorizer__kernel': ['linear', 'rbf'], 'vectorizer__C': [1, 10]}
    parameters = {'kernel': ['linear'], 'C': [1, 10]}

    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    return model, parameters


def train(args):
    data_train = pd.read_csv(args.train_file).sample(frac=1).reset_index(drop=True)
    # data_train.
    # data_dev = pd.read_csv(args.dev_file).sample(frac=1).reset_index(drop=True)
    # data_merge = pd.concat([data_train, data_dev])
    # dev_fold = [-1]*len(data_train) + [0]*len(data_dev)
    # x_traindev, y_traindev = data_merge["text"].to_numpy(), data_merge["label"].to_numpy()
    x_traindev, y_traindev = np.array(data_train["Date"]), np.array(data_train["Open"])
    X = np.vstack((x_traindev, y_traindev)).T
    y_traindev = LabelEncoder().fit_transform(y_traindev)

    pipeline, parameters = build_cls(args.ml_cls, args.tfidf, args.use_hash, args.scaler)
    # pipeline, parameters = build_cls("MLP", True, False, True)
    # print(pipeline.get_params().keys())

    print("- Train the baseline...")
    start = time.time()
    # model = GridSearchCV(pipeline, parameters, cv=PredefinedSplit(test_fold=dev_fold),
    #                      verbose=5,  scoring='f1_weighted')
    model = GridSearchCV(pipeline, parameters, cv=2, verbose=5, scoring='f1_weighted')
    model.fit(X, y_traindev)
    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    best_model = model.best_estimator_
    save(best_model, args.model_name)
    return model


def save(model, mfile):
    print("- Save the model...")
    pkl_write(model, mfile)
    print("\t+ Done.")


def load(mfile):
    print("- Load the model...")
    model = pkl_read(mfile)
    print("\t+ Done.")
    return model


def evaluate(data, model_name):
    model = load(model_name)
    print("- Evaluate the baseline...")
    start = time.time()
    X_dev, y_true = data
    y_pred = model.predict(X_dev)
    mtrcs = class_metrics(y_true, y_pred)
    plot_confusion_matrix(model, X_dev, y_true)
    plot_roc_curve(model, X_dev, y_true)
    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    return mtrcs


def test(args, model_name):
    data_test = pd.read_csv(args.test_file).sample(frac=1).reset_index(drop=True)
    x_test, y_test = data_test["Date"].to_numpy(), data_test["Open"].to_numpy()
    mtrcs = evaluate([x_test, y_test], model_name)
    return mtrcs


def class_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
    f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print("\t+ Accuracy: %.4f(%%)" % (acc * 100))
    measures = {"acc": acc, "prf_macro": f1_ma, "prf_weighted": f1_we, "prf_individual": f1_no}
    return measures


def predict(sent, model_name):
    model = load(model_name)
    label = model.predict([sent]).tolist()[0]
    prob = model.predict_proba([sent]).max()
    print("- Inference...")
    print("\t+ %s with p=%.4f" % (label, prob))
    return label, prob


if __name__ == '__main__':
    """
    python baselines.py --train_file /media/data/langID/small_scale/train.csv --dev_file /media/data/langID/small_scale/dev.csv --test_file /media/data/langID/small_scale/test.csv --model_name ./results/small.NB.m --ml_cls NB
    """
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train_file', help='Trained file', default="data/dji/train.csv", type=str)

    argparser.add_argument('--dev_file', help='Developed file', default="data/dji/test.csv", type=str)

    argparser.add_argument('--test_file', help='Tested file', default="data/dji/test.csv", type=str)

    argparser.add_argument("--tfidf", action='store_true', default=True, help="tfidf flag")

    argparser.add_argument("--use_hash", action='store_true', default=False, help="hashing flag")

    argparser.add_argument("--scaler", action='store_true', default=False, help="scale flag")

    argparser.add_argument('--ml_cls', help='Machine learning classifier', default="SVM", type=str)

    argparser.add_argument('--model_dir', help='Model dir', default="data/dji/", type=str)

    args = argparser.parse_args()

    model_dir, _ = os.path.split(args.model_dir)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_name = os.path.join(args.model_dir, args.ml_cls + ".pickle")

    model = train(args)
    # measures = test(args, args.model_name)
    # label, prob = predict("call us to win a price", args.model_name)
