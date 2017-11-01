 # Keras regressior 
import numpy as np
np.random.seed(1337)  # For reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt  # Import the visualization part

# Create some data
X = np.linspace(-1,1,200)
np.random.shuffle(X)  # Randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
# Plot the data
plt.scatter(X, Y)
plt.show()

# Take the first 160 datas as training data
X_train, Y_train = X[: 160], Y[: 160]
# Take the last 40 datas as testing data
X_test, Y_test = X[160:], Y[160:]

