import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import scipy.io as sp

data = sp.loadmat("hw02_dataset.mat")

X = data["X_trn"]
Y = data["Y_trn"]

print(X, Y)

X_test = data["X_tst"]
Y_test = data["Y_tst"]

# Fit the data to a logistic regression model.
model = linear_model.LogisticRegression()
model.fit(X, [item for sublist in Y for item in sublist])

# Retrieve the model parameters.
b = model.intercept_[0]
w1, w2 = model.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# predict classes
yhat = model.predict(X)

# evaluate the predictions
acc_train = metrics.accuracy_score(Y, yhat)
print("Training Data Classification Accuracy: %.3f" % acc_train)
print()

acc_test = model.score(X_test, Y_test)

print("Testing Data Classification Accuracy: " + str(acc_test))
print("Testing Data Classification Error: " + str(1 - acc_test))

#Configuring two plots
fig, axs = plt.subplots(2, figsize=(6.4, 9.6))
fig.suptitle('Logistic Regression Decision Boundaries')

# Plot the data and the classification with the decision boundary.
xmin, xmax = -2, 2
ymin, ymax = -1, 6
xd = np.array([xmin, xmax])
yd = m*xd + c
axs[0].plot(xd, yd, 'k', lw=1, ls='--')
axs[0].fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
axs[0].fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
axs[1].plot(xd, yd, 'k', lw=1, ls='--')
axs[1].fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
axs[1].fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

def create_axes(X, Y):

    x1_class_0 = []
    x2_class_0 = []
    x1_class_1 = []
    x2_class_1 = []

    for i in range(0, len(X)):
        if(Y[i] == 0):
            x1_class_0.append(X[i][0])
            x2_class_0.append(X[i][1])
        else:
            x1_class_1.append(X[i][0])
            x2_class_1.append(X[i][1])

    return x1_class_0, x2_class_0, x1_class_1, x2_class_1

train_x1_class_0, train_x2_class_0, train_x1_class_1, train_x2_class_1 = create_axes(X, Y)

test_x1_class_0, test_x2_class_0, test_x1_class_1, test_x2_class_1 = create_axes(X_test, Y_test)

axs[0].set_title("Training Data")
axs[0].scatter(train_x1_class_0, train_x2_class_0, s=8, alpha=0.5)
axs[0].scatter(train_x1_class_1, train_x2_class_1, s=8, alpha=0.5)
axs[0].set_xlim([xmin, xmax])
axs[0].set_ylim([ymin, ymax])

axs[1].set_title("Testing Data")
axs[1].scatter(test_x1_class_0, test_x2_class_0, s=8, alpha=0.5)
axs[1].scatter(test_x1_class_1, test_x2_class_1, s=8, alpha=0.5)
axs[1].set_xlim([xmin, xmax])
axs[1].set_ylim([ymin, ymax])

plt.show()



