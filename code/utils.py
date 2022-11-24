from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pandas as pd

def create_pipe(cls,tfidf,**kwargs):
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

        if 'fixed_params' in classifier:
            fixed_params=classifier['fixed_params']
        else:
            fixed_params={}

        if 'float_params' in classifier:
            float_params=classifier['float_params']
            model=create_pipe(classifier['cls'],tfidf=tfidf,**fixed_params)
            cv=RandomizedSearchCV(model,
                             float_params,cv=5,n_jobs=-1,verbose=4,n_iter=30,
                                  scoring=['roc_auc','f1'],refit='roc_auc',random_state=123)
            cv.fit(X,y)
#             if show_features:
#                 n='tvec' if tfidf else 'cvec'
#                 model.named_steps[n].get_feature_names()
            print("Best parameters and accuracy")
            print(cv.best_params_)
            print(f"ROC AUC with CV=5: {cv.best_score_}")
#             return model
        else:
            model=create_pipe(classifier['cls'],tfidf=tfidf,**fixed_params)
            print(f"ROC AUC with CV=5: {cross_val_score(model,X,y,cv=5,n_jobs=-1,scoring='roc_auc').mean()}")
#             return model

        
def get_conf_matrix(model,X,y,random_state=123):
    df_list=[]
    
    for train_index, test_index in KFold(n_splits=5).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        train_accuracy=model.score(X_train,y_train)
        test_accuracy=model.score(X_test,y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(tn,fp,fn,tp)
        prec=tp/(tp+fp)
        recall=tp/(tp+fn)
        spec=tn/(tn+fp)
        f1=2*(prec*recall)/(prec+recall)        
        
        df_list.append([train_accuracy, test_accuracy,prec,recall,spec,f1])
    combined_df=pd.DataFrame(df_list,columns=["Train acc","Test acc","Test Precision","Test Recall","Test Specificity","Test F1"])
    mean_df=combined_df.mean(axis=0).to_frame().T
    return {
            'Train acc':mean_df.iloc[0,0],
            'Test acc':mean_df.iloc[0,1],
            'Test Precision':mean_df.iloc[0,2],
            'Test Recall':mean_df.iloc[0,3],
            'Test Specificity':mean_df.iloc[0,4],
            'Test F1':mean_df.iloc[0,5],            
        }
    