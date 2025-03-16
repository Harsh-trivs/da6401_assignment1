## Setup Instruction :

- Clone the repository
    
    ```sql
    git clone https://github.com/Harsh-trivs/da6401_assignment1.git
    ```
    
- Create a virtual environment
    
    ```sql
    python -m venv venv
    ```
    
    Activate virtual environment
    
    ```powershell
    .\venv\Scripts\activate # Windows
    source venv/bin/activate # Mac/Linux
    ```
    
- Install dependencies
    
    ```powershell
    pip install -r requirements.txt
    ```
    

## Python files used for experiments

- wandbSweep.py
    
    Used for logging experiments on wandb with given hyper-parameters. On execution will carry hyper-parameter sweep and log desired values for further examination.
    
- confusionMatrix.py
    - Used for creation of confusion matrix with most optimal hyper-parameter combination obtained with mean squared loss function and cross entropy function.
    - When executed
        
        ```powershell
        python confusionMatrix.py --lf "cross-entropy"
        ```
        
        generates an run in project da6401_assignment_1 with confusion matrix logged for each epoch. Arg lf help conditionally manage optimal hyper-parameter for cross entropy.
        
        ```powershell
        python confusionMatrix.py --lf "mean_squared_error"
        ```
        
        generates an run in project da6401_assignment_1 with confusion matrix logged for each epoch for mean squared error loss function.
        
- train.py
    
    Used to generate run in given project and entity, while execution supports following parameters 
    
    | Name | Default Value | Description |
    | --- | --- | --- |
    | `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
    | `-we`, `--wandb_entity` | myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
    | `-d`, `--dataset` | fashion_mnist | choices: ["mnist", "fashion_mnist"] |
    | `-e`, `--epochs` | 10 | Number of epochs to train neural network. |
    | `-b`, `--batch_size` | 32 | Batch size used to train neural network. |
    | `-l`, `--loss` | cross_entropy | choices: ["mean_squared_error", "cross_entropy"] |
    | `-o`, `--optimizer` | nadam | choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
    | `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters |
    | `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
    | `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer |
    | `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. |
    | `-beta2`, `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
    | `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
    | `-w_d`, `--weight_decay` | .0005 | Weight decay used by optimizers. |
    | `-w_i`, `--weight_init` | random | choices: ["random", "Xavier"] |
    | `-nhl`, `--num_layers` | 4 | Number of hidden layers used in feedforward neural network. |
    | `-sz`, `--hidden_size` | 256 | Number of hidden neurons in a feedforward layer. |
    | `-a`, `--activation` | tanh | choices: ["identity", "sigmoid", "tanh", "ReLU"] |
