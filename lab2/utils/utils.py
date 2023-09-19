import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy_trends(results, model_name='EEGNet'):
    epochs = range(1, len(results['ReLU'][model_name + '_train']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results['ReLU'][model_name + '_train'], label='relu_train')
    plt.plot(epochs, results['ReLU'][model_name + '_test'], label='relu_test')
    plt.plot(epochs, results['LeakyReLU'][model_name + '_train'], label='leakly_relu_train')
    plt.plot(epochs, results['LeakyReLU'][model_name + '_test'], label='leakly_relu_test')
    plt.plot(epochs, results['ELU'][model_name + '_train'], label='elu_train')
    plt.plot(epochs, results['ELU'][model_name + '_test'], label='elu_test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Acvation function comparison(' + model_name + ')')
    plt.legend()
    plt.grid(True)
    plt.savefig('activation_' + model_name + '.png')
    plt.show()

# plot table of best accuracy for each model and activation function
def plot_table(results):
    activations = ['ReLU', 'LeakyReLU', 'ELU']
    df = pd.DataFrame(columns=['EEGNet', 'DeepConvNet'], index=['ReLU', 'LeakyReLU', 'ELU'])
    for act in activations:
        df.loc[act]['EEGNet'] = max(results[act]['EEGNet_test'])
        df.loc[act]['DeepConvNet'] = max(results[act]['DeepConvNet_test'])
    print(df.transpose())
