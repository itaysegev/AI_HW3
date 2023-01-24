from ID3 import ID3
from utils import *

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


# ========================================================================
def basic_experiment(x_train, y_train, x_test, y_test, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """

    # TODO:
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.

    acc = None

    # ====== YOUR CODE: ======
    tree_instance = ID3(attributes_names)
    tree_instance.fit(x_train, y_train)
    y_pred = tree_instance.predict(x_test)
    acc = accuracy(y_test, y_pred)
    # ========================

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


# ========================================================================

def best_m_test(x_train, y_train, x_test, y_test, min_for_pruning):
    """
        Test the pruning for the best M value we have got from the cross validation experiment.
        :param: best_m: the value of M with the highest mean accuracy across folds
        :return: acc: the accuracy value of ID3 decision tree instance that using the best_m as the pruning parameter.
    """

    # TODO:
    #  - Instate ID3 decision tree instance (using pre-training pruning condition).
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and return the result.

    acc = None

    # ====== YOUR CODE: ======
    tree_instance = ID3(attributes_names, min_for_pruning=50)
    tree_instance.fit(x_train, y_train)
    y_pred = tree_instance.predict(x_test)
    acc = accuracy(y_test, y_pred)
    # ========================

    return acc


def cross_validation_experiment(x, y, plot_graph=True):
    m_lst = [0, 30, 50, 80, 100, 120, 200]
    kf = KFold(n_splits=5, random_state=318866068, shuffle=True)
    avg_lst = []
    for m in m_lst:
        acc_lst = []
        for (train_index, test_index) in kf.split(x, y):
            tree_instance = ID3(attributes_names, min_for_pruning=m)
            tree_instance.fit(x[train_index], y[train_index])
            y_pred = tree_instance.predict(x[test_index])
            acc = accuracy(y[test_index], y_pred)
            acc_lst.append(acc)
        print(acc_lst)
        avg_lst.append(sum(acc_lst) / len(acc_lst))
    print(np.array(avg_lst))
    best_m = m_lst[np.array(avg_lst).argmax()]
    util_plot_graph(m_lst, avg_lst, "M value", "Mean Accuracy", num_folds=len(m_lst))
    return best_m





# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)

    """
    Usages helper:
    (*) To get the results in “informal” or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, formatted_print)

    # """
    #    cross validation experiment
    #    (*) To run the cross validation experiment over the  M pruning hyper-parameter
    #        uncomment below code and run it
    #        modify the value from False to True to plot the experiment result

    # plot_graphs = True
    # best_m = cross_validation_experiment(data_split[0], data_split[1], plot_graph=plot_graphs)
    # print(f'best_m = {best_m}')


    #     pruning experiment, run with the best parameter
    #     (*) To run the experiment uncomment below code and run it
    best_m = 50
    acc = best_m_test(*data_split, min_for_pruning=best_m)
    assert acc > 0.95, 'you should get an accuracy of at least 95% for the pruned ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)
