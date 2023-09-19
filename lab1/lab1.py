import sys
import numpy as np
from utils import generate_linear, generate_XOR_easy, show_result
from model import Model
import time

def train_test_split(x, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    test_indices = indices[:int(test_size * len(x))]
    train_indices = indices[int(test_size * len(x)):]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    if len(sys.argv) != 9:
        print("Usage: python3 lab0.py <linear/xor> <hidden_units> <activation> <epoch> <batchsize> <learning_rate> <loss_function>")
        sys.exit(1)

    if sys.argv[1] == 'linear':
        x, y = generate_linear(100)
    elif sys.argv[1] == 'xor':
        x, y = generate_XOR_easy()

    # x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    hidden_units = int(sys.argv[2])
    activation = sys.argv[3]
    epoch = int(sys.argv[4])
    batchsize = int(sys.argv[5])
    optimizer = sys.argv[6]
    learning_rate = float(sys.argv[7])
    loss_function = sys.argv[8]
    

    model = Model(input_dim=2, hidden_units1=hidden_units, hidden_units2=hidden_units, output_dim=1, activation=activation)
    start = time.time()
    model.train(x, y, epoch, batchsize, learning_rate, loss_function, optimizer)
    end = time.time()
    print('Training time: %.4f' % (end - start))
    model.plot_loss_curve()

    y_pred_list = []
    loss_average = 0
    print("Test Predictions:")
    for xi, y_true in zip(x, y):
        y_pred, y_hat, loss = model.predict(xi.reshape(1, -1), y_true.reshape(1, -1))
        print("Input: {}, Predicted: {:.4f}, Ground Truth: {}".format(xi, y_hat[0][0], y_true[0]))
        y_pred_list.append(y_pred[0][0])
        loss_average += loss
    accuracy = np.mean(y_pred_list == y.squeeze()) * 100
    loss_average /= len(y)
    print("Loss: {:.4f} Accuracy: {:.4f}%".format(loss_average, accuracy))
    show_result(x, y, y_pred_list)

        