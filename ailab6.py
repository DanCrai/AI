import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


# load training data
training_data = np.load('training_data.npy')
prices = np.load('prices.npy')

# shuffle
training_data, prices = shuffle(training_data, prices, random_state=0)


# 3 - fold => 3 samples per fold
num_samples_fold = len(training_data) // 3

# Split train in 3 folds
training_data_1, prices_1 = training_data[:num_samples_fold], \
                            prices[:num_samples_fold]

training_data_2, prices_2 = training_data[num_samples_fold: 2 * num_samples_fold], \
                            prices[num_samples_fold: 2 * num_samples_fold]

training_data_3, prices_3 = training_data[2 * num_samples_fold:], \
                            prices[2 * num_samples_fold:]


def normalize_data(train_data, test_data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)
    scaler_train = scaler.transform(train_data)
    scaler_test = scaler.transform(test_data)
    return (scaler_train, scaler_test)

def avgValue(train_data, train_labels, test_data, test_labels, model):
    norm_data = normalize_data(train_data, test_data)
    norm_train = norm_data[0]
    norm_test = norm_data[1]
    regres = model.fit(norm_train, train_labels)
    predict = regres.predict(norm_test)
    mae = mean_absolute_error(test_labels, predict)
    mse = mean_squared_error(test_labels, predict)
    #print("Mae: ")
    #print(mae)
    #print("Mse: ")
    #print(mse)
    return (mae, mse)
model = LinearRegression()
mae3, mse3 = avgValue(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)), training_data_3, prices_3, model)
mae2, mse2 = avgValue(np.concatenate((training_data_1, training_data_3)), np.concatenate((prices_1, prices_3)), training_data_2, prices_2, model)
mae1, mse1 = avgValue(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)), training_data_1, prices_1, model)

avgMae = (mae1 + mae2 + mae3) / 3
avgMse = (mse1 + mse2 + mse3) / 3

print("avgMae: ")
print(avgMae)
print("avgMse: ")
print(avgMse)

bestAlpha = 0
bestAlphaValue = 9999
for Alpha in [1, 10, 100, 1000]:
    model = Ridge(alpha=Alpha)
    mae3, mse3 = avgValue(np.concatenate((training_data_1, training_data_2)), np.concatenate((prices_1, prices_2)), training_data_3, prices_3, model)
    mae2, mse2 = avgValue(np.concatenate((training_data_1, training_data_3)), np.concatenate((prices_1, prices_3)), training_data_2, prices_2, model)
    mae1, mse1 = avgValue(np.concatenate((training_data_2, training_data_3)), np.concatenate((prices_2, prices_3)), training_data_1, prices_1, model)

    avgMae = (mae1 + mae2 + mae3) / 3
    avgMse = (mse1 + mse2 + mse3) / 3
    if(avgMae + avgMse < bestAlphaValue):
        bestAlpha = Alpha
        bestAlphaValue = avgMae + avgMse
    print("For Alpha = %f" % Alpha)
    print("avgMae: ")
    print(avgMae)
    print("avgMse: ")
    print(avgMse)
print("Best Alpha: %f" % bestAlpha)

print("Ridge regression best performs with alpha = 10.\n")
print("Rapport for ridge regression of parameter alpha = 10: \n")

model = Ridge(bestAlpha)
scaler = preprocessing.StandardScaler()
scaler.fit(training_data)
norm_train = scaler.transform(training_data)
model.fit(norm_train, prices)
print("Coef: \n", model.coef_, "\n")
print("Bias: \n", model.intercept_, "\n")
features = ["Year", "Kilometers Driven", "Fuel_Type", "Transmission", "Owner_Type", "Mileage", "Engine", "Power"]
most_significant_feature = features[int(np.argmax(np.abs(model.coef_)))]
second_most_significant_feature = features[int(np.argmax(np.abs(model.coef_))) + 1]
print("Most significant feature: %s" % most_significant_feature)
print("Second most significant feature: %s" % second_most_significant_feature)
