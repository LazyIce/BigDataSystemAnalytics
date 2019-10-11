import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier


RANDOM_STATE = 6220


def loadData():
    data = pd.read_csv("../data/sky.csv")
    print(data.info())
    print(data.describe())
    print(list(data.isnull().any()))

    return data


def preprocess_data(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1:].values

    labelEncoder = LabelEncoder()
    Y = labelEncoder.fit_transform(np.ravel(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=RANDOM_STATE, test_size=0.2)
    mm = MinMaxScaler()
    X_train_scaled = mm.fit_transform(X_train)
    X_test_scaled = mm.fit_transform(X_test)

    return X_train_scaled, X_test_scaled, Y_train, Y_test
    

def select_lr(X, Y):
    grid_param_1 = ('newton-cg', 'lbfgs', 'sag', 'saga')
    grid_param_2 = range(200, 1010, 10)
    parameters = {'solver': grid_param_1, 'max_iter': grid_param_2}
    lr = LogisticRegression(multi_class='multinomial', random_state=RANDOM_STATE)
    clf = GridSearchCV(lr, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('logistic regression', mean_scores, grid_param_1, grid_param_2, 'solver', 'max_iter')
    print('The best parameters for lr: ' + str(clf.best_params_))
    print('The best accuracy for lr: ' + str(clf.best_score_))

    return clf.best_estimator_


def select_knn(X, Y):
    grid_param_1 = ('uniform', 'distance')
    grid_param_2 = range(3, 51)
    parameters = {'weights': grid_param_1, 'n_neighbors': grid_param_2}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('k-nearest neighbors', mean_scores, grid_param_1, grid_param_2, 'weights', 'n_neighbors')
    print('The best parameters for knn: ' + str(clf.best_params_))
    print('The best accuracy for knn: ' + str(clf.best_score_))

    return clf.best_estimator_


def select_svm(X, Y):
    grid_param_1 = ('linear', 'poly', 'rbf', 'sigmoid')
    grid_param_2 = [0.001, 0.01, 0.1, 1, 10]
    parameters = {'kernel': grid_param_1, 'C': grid_param_2}
    SVM = svm.SVC(gamma='auto', random_state=RANDOM_STATE)
    clf = GridSearchCV(SVM, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('supported vector machine', mean_scores, grid_param_1, grid_param_2, 'kernel', 'penalty_C')
    print('The best parameters for svm: ' + str(clf.best_params_))
    print('The best accuracy for svm: ' + str(clf.best_score_))

    return clf.best_estimator_


def select_dt(X, Y):
    grid_param_1 = ('gini', 'entropy')
    grid_param_2 = range(3, 51)
    parameters = {'criterion': grid_param_1, 'max_depth': grid_param_2}
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf = GridSearchCV(dt, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('decision tree', mean_scores, grid_param_1, grid_param_2, 'criterion', 'max_depth')
    print('The best parameters for dt: ' + str(clf.best_params_))
    print('The best accuracy for dt: ' + str(clf.best_score_))

    return clf.best_estimator_


def select_mlp(X, Y):
    grid_param_1 = [(5, ), (5, 10), (5, 10, 15), (20, )]
    grid_param_2 = range(200, 810, 10)
    parameters = {'hidden_layer_sizes': grid_param_1, 'max_iter': grid_param_2}
    mlp = MLPClassifier(learning_rate_init=0.1, learning_rate='adaptive', random_state=RANDOM_STATE)
    clf = GridSearchCV(mlp, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('multi layer perceptron', mean_scores, grid_param_1, grid_param_2, 'hidden_layer_sizes', 'max_iter')
    print('The best parameters for mlp: ' + str(clf.best_params_))
    print('The best accuracy for mlp: ' + str(clf.best_score_))

    return clf.best_estimator_


def select_bc(X, Y, estimator):
    grid_param_1 = (True, False)
    grid_param_2 = range(5, 101, 5)
    parameters = {'bootstrap': grid_param_1, 'n_estimators': grid_param_2}
    bc = BaggingClassifier(estimator, random_state=RANDOM_STATE)
    clf = GridSearchCV(bc, parameters, cv=5, scoring='accuracy')
    clf.fit(X, Y)
    cv_results = clf.cv_results_
    mean_scores = cv_results['mean_test_score']
    plot_grid_search('bagging classifier', mean_scores, grid_param_1, grid_param_2, 'bootstrap', 'n_estimators')
    print('The best parameters for bc: ' + str(clf.best_params_))
    print('The best accuracy for bc: ' + str(clf.best_score_))

    return clf.best_estimator_


def plot_grid_search(title, mean_scores, grid_param_1, grid_param_2, name_param_1, name_param_2):
    
    mean_scores = np.array(mean_scores).reshape(len(grid_param_1),len(grid_param_2))

    fig, ax = plt.subplots(1,1)

    for idx, val in enumerate(grid_param_1):
        ax.plot(grid_param_2, mean_scores[idx,:], label= name_param_1 + ': ' + str(val))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=12)
    ax.set_ylabel('CV Mean Accuracy', fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid('on')
    plt.tight_layout()
    fig.savefig('../img/' + '_'.join(title.split(' ')) + '.png')


def run_test(name, estimator, X, Y):
    Y_pred = estimator.predict(X)
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred, average='macro')
    recall = recall_score(Y, Y_pred, average='macro')
    f1score = f1_score(Y, Y_pred, average='macro')
    print('The quality metrics for the ' + name + 'classifier are as follows:')
    print('accuracy: ' + str(accuracy))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('f1_score: ' + str(f1score))


def main():
    print('********** load data start **********')
    data = loadData()
    print('********** load data end **********')

    print('********** preprocess data start **********')
    X_train, X_test, Y_train, Y_test = preprocess_data(data)
    print('********** preprocess data end **********')

    print('********** select models start **********')
    best_lr = select_lr(X_train, Y_train)
    best_knn = select_knn(X_train, Y_train)
    best_svm = select_svm(X_train, Y_train)
    best_dt = select_dt(X_train, Y_train)
    best_mlp = select_mlp(X_train, Y_train)
    best_bc = select_bc(X_train, Y_train, best_dt)
    print('********** select models end **********')

    print('********** run test data start **********')
    run_test('logistic regression', best_lr, X_test, Y_test)
    run_test('k-nearest neighbors', best_knn, X_test, Y_test)
    run_test('supported vector machines', best_svm, X_test, Y_test)
    run_test('decision tree', best_dt, X_test, Y_test)
    run_test('multi-layer perceptron', best_mlp, X_test, Y_test)
    run_test('bagging classifier', best_bc, X_test, Y_test)
    print('********** run test data end **********')


if __name__ == "__main__":
    main()
