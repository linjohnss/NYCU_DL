import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, input_dim, hidden_units1, hidden_units2, output_dim, activation):
        self.input_dim = input_dim
        self.hidden_units1 = hidden_units1
        self.hidden_units2 = hidden_units2
        self.output_dim = output_dim

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_units1)
        self.b1 = np.random.randn(1, hidden_units1)
        self.W2 = np.random.randn(hidden_units1, hidden_units2)
        self.b2 = np.random.randn(1, hidden_units2)
        self.W3 = np.random.randn(hidden_units2, output_dim)
        self.b3 = np.random.randn(1, output_dim)

        # Store the training loss
        self.losses = []

        # Activation function
        self.without_activation = False
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.derivate_activation = self.derivate_sigmoid
        elif activation == 'relu':
            self.activation = self.relu
            self.derivate_activation = self.derivate_relu
        elif activation == 'none':
            self.without_activation = True
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivate_sigmoid(self, x):
        return np.multiply(x, 1.0 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def derivate_relu(self, x):
        return np.where(x > 0, 1, 0)

    def l2_loss(self, y, y_hat):
        return np.mean(np.square(y - y_hat)) / 2

    def derivate_l2_loss(self, y, y_hat):
        return y_hat - y

    def binary_cross_entropy(self, y, y_hat):
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def derivate_binary_cross_entropy(self, y, y_hat):
        return (y_hat - y) / (y_hat * (1 - y_hat))

    def forward_pass(self, X):        
        if self.without_activation:
            # no activation function
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = Z1
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = Z2
            Z3 = np.dot(A2, self.W3) + self.b3
            y_hat = Z3
        else:
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.activation(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.activation(Z2)
            Z3 = np.dot(A2, self.W3) + self.b3
            y_hat = self.sigmoid(Z3)
            
        return y_hat, A2, A1

    def backpropagation(self, X, y, y_hat, A2, A1):
        m = y.shape[0]

        # Calculate the gradients of the output layer
        dL = self.derivate_loss(y, y_hat)
        if self.without_activation:
            dZ3 = dL
        else:
            dZ3 = dL * self.derivate_sigmoid(y_hat)
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m
        
        # Calculate the gradients of the second hidden layer
        dA2 = np.dot(dZ3, self.W3.T)
        if self.without_activation:
            dZ2 = dA2
        else:
            dZ2 = dA2 * self.derivate_activation(A2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        
        # Calculate the gradients of the first hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        if self.without_activation:
            dZ1 = dA1
        else:
            dZ1 = dA1 * self.derivate_activation(A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def train(self, X, y, epochs, batchsize = 10, learning_rate = 0.1, loss_function = 'l2_loss', optimizer = 'SGD'):
        # Determine loss function
        if loss_function == 'l2_loss':
            self.loss_function = self.l2_loss
            self.derivate_loss = self.derivate_l2_loss
        elif loss_function == 'binary_cross_entropy':
            self.loss_function = self.binary_cross_entropy
            self.derivate_loss = self.derivate_binary_cross_entropy
        # Start training
        for epoch in range(epochs):
            # initialize loss for each epoch
            loss_batch = 0
            for step in range(0, X.shape[0], batchsize):
                X_batch = X[step:step+batchsize]
                y_batch = y[step:step+batchsize]
                # Forward pass
                y_hat, A2, A1 = self.forward_pass(X_batch)

                # Calculate loss
                loss = self.loss_function(y_batch, y_hat)
                loss_batch += loss

                # Backpropagation
                dW1, db1, dW2, db2, dW3, db3 = self.backpropagation(X_batch, y_batch, y_hat, A2, A1)

                if optimizer == 'SGD':
                    # Update weights and biases (Stochastic Gradient Descent)
                    self.W1 -= learning_rate * dW1
                    self.b1 -= learning_rate * db1
                    self.W2 -= learning_rate * dW2
                    self.b2 -= learning_rate * db2
                    self.W3 -= learning_rate * dW3
                    self.b3 -= learning_rate * db3
                elif optimizer == 'Momentum':
                    # Update weights and biases (Momentum)
                    # Set initial velocity to 0
                    if epoch == 0:
                        self.vW1 = np.zeros_like(self.W1)
                        self.vb1 = np.zeros_like(self.b1)
                        self.vW2 = np.zeros_like(self.W2)
                        self.vb2 = np.zeros_like(self.b2)
                        self.vW3 = np.zeros_like(self.W3)
                        self.vb3 = np.zeros_like(self.b3)
                    self.vW1 = 0.9 * self.vW1 + learning_rate * dW1
                    self.vb1 = 0.9 * self.vb1 + learning_rate * db1
                    self.vW2 = 0.9 * self.vW2 + learning_rate * dW2
                    self.vb2 = 0.9 * self.vb2 + learning_rate * db2
                    self.vW3 = 0.9 * self.vW3 + learning_rate * dW3
                    self.vb3 = 0.9 * self.vb3 + learning_rate * db3
                    self.W1 -= self.vW1
                    self.b1 -= self.vb1
                    self.W2 -= self.vW2
                    self.b2 -= self.vb2
                    self.W3 -= self.vW3
                    self.b3 -= self.vb3
                elif optimizer == 'Adam':
                    # Update weights and biases (Adam)
                    # Set initial velocity and squared gradient to 0
                    if epoch == 0:
                        self.vW1 = np.zeros_like(self.W1)
                        self.vb1 = np.zeros_like(self.b1)
                        self.vW2 = np.zeros_like(self.W2)
                        self.vb2 = np.zeros_like(self.b2)
                        self.vW3 = np.zeros_like(self.W3)
                        self.vb3 = np.zeros_like(self.b3)
                        self.sW1 = np.zeros_like(self.W1)
                        self.sb1 = np.zeros_like(self.b1)
                        self.sW2 = np.zeros_like(self.W2)
                        self.sb2 = np.zeros_like(self.b2)
                        self.sW3 = np.zeros_like(self.W3)
                        self.sb3 = np.zeros_like(self.b3)
                    self.vW1 = 0.9 * self.vW1 + (1 - 0.9) * dW1
                    self.vb1 = 0.9 * self.vb1 + (1 - 0.9) * db1
                    self.vW2 = 0.9 * self.vW2 + (1 - 0.9) * dW2
                    self.vb2 = 0.9 * self.vb2 + (1 - 0.9) * db2
                    self.vW3 = 0.9 * self.vW3 + (1 - 0.9) * dW3
                    self.vb3 = 0.9 * self.vb3 + (1 - 0.9) * db3
                    self.sW1 = 0.999 * self.sW1 + (1 - 0.999) * np.square(dW1)
                    self.sb1 = 0.999 * self.sb1 + (1 - 0.999) * np.square(db1)
                    self.sW2 = 0.999 * self.sW2 + (1 - 0.999) * np.square(dW2)
                    self.sb2 = 0.999 * self.sb2 + (1 - 0.999) * np.square(db2)
                    self.sW3 = 0.999 * self.sW3 + (1 - 0.999) * np.square(dW3)
                    self.sb3 = 0.999 * self.sb3 + (1 - 0.999) * np.square(db3)
                    self.W1 -= learning_rate * self.vW1 / (np.sqrt(self.sW1) + 1e-8)
                    self.b1 -= learning_rate * self.vb1 / (np.sqrt(self.sb1) + 1e-8)
                    self.W2 -= learning_rate * self.vW2 / (np.sqrt(self.sW2) + 1e-8)
                    self.b2 -= learning_rate * self.vb2 / (np.sqrt(self.sb2) + 1e-8)
                    self.W3 -= learning_rate * self.vW3 / (np.sqrt(self.sW3) + 1e-8)
                    self.b3 -= learning_rate * self.vb3 / (np.sqrt(self.sb3) + 1e-8)
                print('Step %d, loss=%.4f' % (step, loss), end='\r')
            self.losses.append(loss_batch / (step + 1))
            if (epoch+1) % 100 == 0:
                print('Epoch %d/%d, loss=%.4f' % (epoch + 1, epochs, loss))

    def predict(self, X, y):
        y_hat, _ , _ = self.forward_pass(X)
        y_pred = np.round(y_hat)
        loss = self.loss_function(y, y_hat)
        return y_pred, y_hat, loss
    
    def plot_loss_curve(self):
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_curve.png')
        plt.show()