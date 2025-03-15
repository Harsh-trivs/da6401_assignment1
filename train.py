import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tensorflow.keras.datasets import mnist, fashion_mnist
from model import FeedForwardNN  # Assuming your model is saved as model.py

# Argument parser
def get_args():
    parser = argparse.ArgumentParser(description="Train a Neural Network")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", default="myname")
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init",choices= ["random", "Xavier"],default='Xavier')
    parser.add_argument("-nhl", "--num_layers", type=int, default=4)
    parser.add_argument("-sz", "--hidden_size", type=int, default=256)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh")
    return parser.parse_args()

# Dataset loading
def load_data(dataset):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

# Main training flow
def main():
    args = get_args()

    X_train, y_train, X_test, y_test = load_data(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
    X_train, X_val, X_test = X_train.reshape(-1, 28 * 28), X_val.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)


    hidden_layers = [(args.hidden_size, args.activation)] * args.num_layers
    model = FeedForwardNN(
                          input_size=784, 
                          hidden_layers=hidden_layers, 
                          output_size=10,
                          lr=args.learning_rate, 
                          optimizer=args.optimizer, 
                          momentum=args.momentum,
                          beta=args.beta,
                          beta1=args.beta1, 
                          beta2=args.beta2, 
                          epsilon=args.epsilon,
                          alpha=args.weight_decay, 
                          batch_size=args.batch_size, 
                          loss_function=args.loss,
                          weight_init = args.weight_init,
                              )

    # Initialize wandb
    wandb.finish()  # Close any existing session
    wandb.init(project=args.wandb_project,name=args.wandb_entity, reinit=True)

    
    for i in range(args.epochs):
        print(f"Training on Fold: {i + 1}")
        model.train(X_train, y_train, epochs=1)

        y_val_pred = model.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)
        y_test_pred = model.predict(X_test)
        test_accuracy = np.mean(y_test_pred == y_test)

        loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)

        cm = confusion_matrix(y_test, y_test_pred)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average=None)
        recall = recall_score(y_test, y_test_pred, average=None)

        display_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=display_labels, yticklabels=display_labels)

        for i in range(len(display_labels)):
            plt.text(len(display_labels) + 1.5, i + 0.5, f'P: {precision[i]:.2f}', va='center', ha='left', fontsize=10, color='black')
            plt.text(i + 0.5, len(display_labels) + 1.2, f'R: {recall[i]:.2f}', va='top', ha='center', fontsize=10, color='black')

        plt.title(f'Confusion Matrix with Accuracy: {accuracy:.2%}')
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.tight_layout()
        # Log the confusion matrix image to WandB
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "loss": loss,
            "test_accuracy": test_accuracy,
            "confusion_matrix": wandb.Image(plt),
        })
        plt.close()

if __name__ == "__main__":
    main()
