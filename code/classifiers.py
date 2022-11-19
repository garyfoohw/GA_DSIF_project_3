from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from statistics import mean
import pandas as pd

class Classifiers():
    def __init__(self,create_pipe,X,y,seed=42):
        self.create_pipe=create_pipe
        self.kf=KFold(n_splits=5,random_state=seed,shuffle=True)
        self.X=X
        self.y=y
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=seed)
        
    def print_report(self,y_pred, name):
        train_accuracy=self.pipe.score(self.X_train,self.y_train)
        test_accuracy=self.pipe.score(self.X_test,self.y_test)
#         print("--Train--")
#         print(f"Accuracy: {train_accuracy:.4f}")
#         print("--Test--")
#         print(f"Accuracy: {test_accuracy:.4f}")    

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        prec=tp/(tp+fp)
        recall=tp/(tp+fn)
        spec=tn/(tn+fp)
        f1=2*(prec*recall)/(prec+recall)
#         print(f"Precision: {prec:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"Specificity: {spec:.4f}")
#         print(f"F1: {f1:.4f}")
        return [train_accuracy, test_accuracy,prec,recall,spec,f1]
        
    def print_metrics(self,name):
        if name == "RandomForest":
            print(f"---Other metrics regarding {name}---")
#             print(self.pipe['cls'].estimators_)
            print(f"Average tree depth: {mean([estimator.get_depth() for estimator in self.pipe['cls'].estimators_])}")
        
        
    def run_classifier(self, cls, name, **kwargs):
        print(f"========= Running classifier: {name} =========")
        self.pipe=self.create_pipe(cls,**kwargs)
        df_list=[]
        
        for train_index, test_index in self.kf.split(self.X):
# ...     print("TRAIN:", train_index, "TEST:", test_index)
            self.X_train, self.X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            self.y_train, self.y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.pipe.fit(self.X_train,self.y_train)
            y_pred = self.pipe.predict(self.X_test)
            df_list.append(self.print_report(y_pred,name))
#             self.print_metrics(name)
        combined_df=pd.DataFrame(df_list,columns=["Train acc","Test acc","Test Precision","Test Recall","Test Specificity","Test F1"])
#         display(combined_df)
#         print("Mean KFold scores:")
        mean_df=combined_df.mean(axis=0).to_frame().T
#         display(mean_df)
        return {
            'Train acc':mean_df.iloc[0,0],
            'Test acc':mean_df.iloc[0,1],
            'Test Precision':mean_df.iloc[0,2],
            'Test Recall':mean_df.iloc[0,3],
            'Test Specificity':mean_df.iloc[0,4],
            'Test F1':mean_df.iloc[0,5],            
        }
        