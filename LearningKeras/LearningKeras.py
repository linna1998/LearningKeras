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

# Build up the model
print('Building-------')
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# Choose the loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# Training the model
print('Training-------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    # After 100 steps, print the train cost out
    if step % 100 == 0:
        print('train cost', cost)
