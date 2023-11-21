from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

if __name__ == '__main__':
    '''
    nb_samples = 500
    X, Y = make_classification(n_samples=nb_samples, n_features=2, n_redundant=0, n_classes=2)

    lr = LogisticRegression()
    svc = SVC(kernel='poly', probability=True)
    dt = DecisionTreeClassifier()
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)

    classifiers = [('lr', lr), ('dt', dt), ('svc', svc)]
    vc = VotingClassifier(estimators=classifiers, voting='hard')

    a = [cross_val_score(lr, X, Y, scoring='accuracy', cv=10).mean(),
         cross_val_score(svc, X, Y, scoring='accuracy', cv=10).mean(),
         cross_val_score(dt, X, Y, scoring='accuracy', cv=10).mean(),
         cross_val_score(ada, X, Y, scoring='accuracy', cv=10).mean(),
         cross_val_score(vc, X, Y, scoring='accuracy', cv=10).mean()]

    print(a)
    '''

    iris = load_iris()
    X = iris.data[:, 2:]
    Y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2)
    tree_clf.fit(X, Y)

    f = open("./iris_tree.dot", 'w')
    export_graphviz(tree_clf,
                    out_file=f,
                    feature_names=iris.feature_names[2:],
                    class_names=iris.target_names,
                    rounded=True,
                    filled=True
                    )