# Lab: Data-Centric vs Model-Centric approaches

# In this lab, we'll explore the difference between focusing on improving the model versus improving the data.
# We will train a classifier for magazine product reviews and evaluate multiple stages:
# - baseline model
# - model-centric improvements
# - data-centric improvements

import pandas as pd
import re

train = pd.read_csv('reviews_train.csv')
test = pd.read_csv('reviews_test.csv')

test.sample(5)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import clone

sgd_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

_ = sgd_clf.fit(train['review'], train['label'])

from sklearn import metrics

def evaluate(clf):
    pred = clf.predict(test['review'])
    acc = metrics.accuracy_score(test['label'], pred)
    print(f'Accuracy: {100*acc:.1f}%')
    
evaluate(sgd_clf)

# Try a MultinomialNB with unigram+bigram tf-idf features for better recall on short phrases
nb_clf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ('clf', MultinomialNB()),
])
_ = nb_clf.fit(train['review'], train['label'])
evaluate(nb_clf)

# Exercise 2: inspect potentially bad data points (e.g., HTML fragments)
html_like = train[train['review'].str.contains(r'<[^>]+|&lt;|&gt;', regex=True, na=False)]
print(f'Potential HTML-ish rows: {len(html_like)} of {len(train)}')
print(html_like[['label', 'review']].head(5))


# Exercise 3: simple heuristic to drop HTML/noisy rows, then retrain
def is_bad_data(review: str) -> bool:
    text = str(review)
    return bool(re.search(r'<[^>]+>|&lt;|&gt;|&#|href=|div>', text))


train_clean = train[~train['review'].map(is_bad_data)]
print(f'Cleaned train size: {len(train_clean)} (removed {len(train) - len(train_clean)})')

sgd_clf_clean = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
_ = sgd_clf_clean.fit(train_clean['review'], train_clean['label'])
evaluate(sgd_clf_clean)

nb_clf_clean = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ('clf', MultinomialNB()),
])
_ = nb_clf_clean.fit(train_clean['review'], train_clean['label'])
evaluate(nb_clf_clean)


# Helper to compare multiple classifiers before/after cleaning
def compare_models(models):
    rows = []
    for name, model in models:
        raw_model = clone(model)
        raw_model.fit(train['review'], train['label'])
        raw_acc = metrics.accuracy_score(test['label'], raw_model.predict(test['review']))

        clean_model = clone(model)
        clean_model.fit(train_clean['review'], train_clean['label'])
        clean_acc = metrics.accuracy_score(test['label'], clean_model.predict(test['review']))

        rows.append({'model': name, 'raw_acc': raw_acc, 'clean_acc': clean_acc})

    df = pd.DataFrame(rows)
    print('\nAccuracy comparison (raw vs cleaned):')
    print(df)
    return df

compare_models([
    ('SGD', sgd_clf),
    ('MultinomialNB', nb_clf),
])
