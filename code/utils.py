from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

def create_pipe(cls,tfidf,**kwargs):
    #create either a CountVectorizer or TfidfVectorizer depending on param
    if tfidf:
        return Pipeline([
            ('tvec',TfidfVectorizer(stop_words=stopwords.words("english"),token_pattern="[^\W\d_]+", **kwargs)),
            ('cls',cls)
        ])
    else:
        return Pipeline([
            ('cvec',CountVectorizer(stop_words=stopwords.words("english"),token_pattern="[^\W\d_]+",**kwargs)),
            ('cls',cls)
        ])


def run_classifiers(classifiers_list,X,y,tfidf=False):
    for classifier in classifiers_list:
        name=classifier['name']
        print(f"======= Running classifier: {name} =======")
        
        #if 'fixed_params' exist, spread the elements into the vectorizer object
        if 'fixed_params' in classifier:
            fixed_params=classifier['fixed_params']
        else:
            fixed_params={}
        
        #if 'float_params' exist, run a Randomized Search using the params
        if 'float_params' in classifier:
            float_params=classifier['float_params']
            model=create_pipe(classifier['cls'],tfidf=tfidf,**fixed_params)
            cv=RandomizedSearchCV(model,
                             float_params,cv=5,n_jobs=-1,verbose=4,n_iter=30,
                                  scoring=['roc_auc','f1'],refit='roc_auc',random_state=123)
            cv.fit(X,y)

            print("Best parameters and accuracy")
            print(cv.best_params_)
            print(f"ROC AUC with CV=5: {cv.best_score_}")
        else:
            model=create_pipe(classifier['cls'],tfidf=tfidf,**fixed_params)
            print(f"ROC AUC with CV=5: {cross_val_score(model,X,y,cv=5,n_jobs=-1,scoring='roc_auc').mean()}")