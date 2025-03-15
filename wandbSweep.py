import argparse
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import numpy as np
from model import FeedForwardNN

def main(args):
    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train,_,y_train,_ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

    # Normalize inputs
    X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

    # Flatten inputs
    X_train, X_val, X_test = X_train.reshape(-1, 28 * 28), X_val.reshape(-1, 28 * 28), X_test.reshape(-1, 28 * 28)


    # Define training function for WandB sweep
    def train():
        wandb.init()
        config = wandb.config

        # Create a descriptive experiment name
        wandb.run.name = (
            f"layers_{config.num_hidden_layers}_"
            f"hls_{config.hidden_layer_size}_"
            f"act_{config.activation_function}_"
            f"lr_{config.learning_rate}_"
            f"opt_{config.optimizer}_"
            f"alpha_{config.alpha}_"
            f"bs_{config.batch_size}"
        )

        # Define the model
        model = FeedForwardNN(
            input_size=28 * 28,
            hidden_layers=[(config.hidden_layer_size, config.activation_function)] * config.num_hidden_layers,
            output_size=10,
            lr=config.learning_rate,
            optimizer=config.optimizer,
            alpha=config.alpha,
            batch_size=config.batch_size,
            loss_function=args.lf
        )

        # Train model
        for epoch in range(config.epochs):
            model.train(X_train, y_train, epochs=1)

            # Compute validation accuracy
            y_pred = model.predict(X_val)
            val_accuracy = np.mean(y_pred == y_val)
            # Log metrics to WandB
            y_test_pred = model.predict(X_test)
            test_accuracy = np.mean(y_test_pred == y_test)

            loss = model.loss(X_train, y_train)
            val_loss = model.loss(X_val, y_val)

            wandb.log({"epoch": epoch + 1, 
                    "loss": loss, 
                    "test_accuracy": test_accuracy,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss})

    # Define WandB sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization for efficiency
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'epochs': {'values': [10, 15]},
            'num_hidden_layers': {'values': [3,4,5]},
            'hidden_layer_size': {'values': [64, 128, 256]},
            'learning_rate': {'values':[1e-3,1e-4]},
            'optimizer': {'values': ['momentum','sgd','rmsprop', 'adam', 'nadam']},
            'activation_function': {'values': ['tanh', 'relu']},
            'alpha': {'values': [0, 0.0005, 0.5]},
            'batch_size': {'values': [32, 64]} 
        }
    }

    # Run the sweep
    sweep_id = wandb.sweep(sweep_config, project="da6401_assignment_1")
    wandb.agent(sweep_id, train, count=75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    # Adding arguments
    parser.add_argument("--lf", type=str, default="cross_entropy", help="Loss function to use")
    args = parser.parse_args()
    main(args)