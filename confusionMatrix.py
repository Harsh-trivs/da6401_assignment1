import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from model import FeedForwardNN
import argparse

def main(args):
    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
    X_train, X_val, X_test = X_train.reshape(-1, 28 * 28), X_val.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)

    # Define the model
    if args.lf == "cross_entropy":
        model = FeedForwardNN(
        input_size=28 * 28,
        hidden_layers=[(256, 'tanh')*4],
        output_size=10,
        lr=0.001,
        optimizer='nadam',
        alpha=0.0005,
        batch_size=32,
        loss_function=args.lf
    )
    else:
        model = FeedForwardNN(
            input_size=28 * 28,
            hidden_layers=[(64, 'relu')*5],
            output_size=10,
            lr=0.001,
            optimizer='nadam',
            alpha=0,
            batch_size=32,
            loss_function=args.lf
    )

    # Initialize wandb
    wandb.finish()  # Close any existing session
    wandb.init(project="da6401_assignment_1_mse",name="confusion_matrix", reinit=True)

    val_acc = []
    val_l = []
    y_preds = []
    epochs = 10 if args.lf == "cross_entropy" else 15
    for i in range(epochs):
        print(f"Training on Fold: {i + 1}")
        model.train(X_train, y_train, epochs=1)

        y_val_pred = model.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)
        y_test_pred = model.predict(X_test)
        test_accuracy = np.mean(y_test_pred == y_test)

        loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)

        val_l.append(val_loss)
        val_acc.append(val_accuracy)
        y_preds.append(y_test_pred)

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

        plt.title(f'Confusion Matrix with Accuracy: {accuracy:.2%} (Epoch {i+1})')
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.tight_layout()

        # Log the confusion matrix image to WandB
        wandb.log({
            "confusion_matrix": wandb.Image(plt),
        })

    print("val loss", val_l)
    print("val acc", val_acc)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    # Adding arguments
    parser.add_argument("--lf", type=str, default="cross_entropy", help="Loss function to use")
    args = parser.parse_args()
    main(args)