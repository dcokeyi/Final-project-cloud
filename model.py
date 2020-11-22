import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.metrics import r2_score


def pkl_write(data, filename='data/dji/data.pickle'):
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pkl_read(filename='data/dji/data.pickle'):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data


def create_model(data):
    df_train = pd.read_csv(data, sep=",", header=0, usecols=[0, 1])
    x_train = np.array(df_train[["Date"]])
    y_train = np.array(df_train[["Open"]])
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    print("coefficients: ", regr.coef_)  # Slope
    print("Intercept: ", regr.intercept_)  # Intercept
    save(regr)
    return regr


def test_model(data, model):
    df_test = pd.read_csv(data, sep=",", header=0, usecols=[0, 1])
    x_test = np.array(df_test[["Date"]])
    y_test = np.array(df_test[["Open"]])
    test_y_ = model.predict(x_test)
    print("Mean absolute error % .2f" % np.mean(np.absolute(test_y_ - y_test)))
    print("Mean sum of squares(MSE): % .2f" % np.mean((test_y_ - y_test) ** 2))
    print("R2 - score: % .2f" % r2_score(test_y_, y_test))


# Function for predicting future values :
def predict(date):
    model = load()
    predicted_value = float(date) * model.coef_[0][0] + model.intercept_[0]
    return predicted_value


def save(model):
    print("- Save the model...")
    pkl_write(model)
    print("\t+ Done.")


def load():
    print("- Loading the model...")
    model = pkl_read()
    print("\t+ Done.")
    return model


#if __name__ == '__main__':
#    train_file = 'data/dji/train.csv'
#    test_file = 'data/dji/test.csv'

#    model = create_model(train_file)
#    test_model(test_file, model)

#    prediction_date = 45000
#    price_prediction = predict(prediction_date)
#    print("Estimated price: ", price_prediction)

def search(prediction_date):
    train_file = 'data/dji/train.csv'
    test_file = 'data/dji/test.csv'

    model = create_model(train_file)
    test_model(test_file, model)
    price_prediction = predict(prediction_date)

    print("Estimated price: ", price_prediction)
    return price_prediction
