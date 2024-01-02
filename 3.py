from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)

    # 预测
    y_pred = tree_clf.predict(X_test) # 使用符号函数将预测结果转换为0和1

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    f = open("./iris_tree.dot", 'w')
    export_graphviz(tree_clf,
                    out_file=f,
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    rounded=True,
                    filled=True
                    )
    f.close()

    source = Source.from_file("./iris_tree.dot")
    source.view()