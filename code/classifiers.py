from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statistics import mean

class Classifiers():
    def __init__(self,create_pipe,X,y,random_seed=42):
        self.create_pipe=create_pipe
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=random_seed)
        
    def print_report(self,y_pred, name):
        print("--Train--")
        print(f"Accuracy: {self.pipe.score(self.X_train,self.y_train):.4f}")
        print("--Test--")
        print(f"Accuracy: {self.pipe.score(self.X_test,self.y_test):.4f}")    

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        prec=tp/(tp+fp)
        recall=tp/(tp+fn)
        spec=tn/(tn+fp)
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"F1: {2*(prec*recall)/(prec+recall):.4f}")
        
    def print_metrics(self,name):
        if name == "Random Forest":
            print(f"---Other metrics regarding {name}---")
#             print(self.pipe['cls'].estimators_)
            print(f"Average tree depth: {mean([estimator.get_depth() for estimator in self.pipe['cls'].estimators_])}")
        
        
    def run_classifier(self, cls, name):
        print(f"========= Running classifier: {name} =========")
        self.pipe=self.create_pipe(cls)
        self.pipe.fit(self.X_train,self.y_train)
        y_pred = self.pipe.predict(self.X_test)
        self.print_report(y_pred,name)
        self.print_metrics(name)
        