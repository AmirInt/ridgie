import sys
import ridgie.ridge_regression_utils as rrutils
import ridgie.dataset_utils as dutils
import ridgie.config_utils as cutils
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def run():
    n_dataset, n_dimension, weight_ones, alpha = cutils.get_config()

    x, y = dutils.generate_dataset(n_dataset, n_dimension, weight_ones)

    w, b, losses = rrutils.ridge_regression_gd(x, y, alpha)

    print("w:")
    print(w)

    print("b:")
    print(b)

    print("Final Loss:", losses[-1])

    plt.plot(losses, 'r')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.show()


def compare():
    n_dataset, n_dimension, weight_ones, alpha = cutils.get_config()

    x, y = dutils.generate_dataset(n_dataset, n_dimension, weight_ones)

    w, b, _ = rrutils.ridge_regression_gd(x, y, alpha)

    regr = linear_model.Ridge(alpha=alpha)
    regr.fit(x, y)

    print("MSE of gradient descent solver: ", rrutils.mse(x, y, w, b))
    print("MSE of built-in solver: ", mean_squared_error(regr.predict(x), y))
    print("Distance between w-coefficients: ", np.linalg.norm(w - regr.coef_))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "run":
            run()
        elif sys.argv[1] == "compare":
            compare()
        else:
            raise IndexError
    except IndexError:
        print("Arguments:")
        print()
        print("- run (Run the manually written model on data)")
        print()
        print("- compare (Run both the manually written and the scikit-learn's built-in model and compare the results)")
