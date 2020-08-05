############# Project Tree-view #############
.
├── README.txt
├── const.py
├── data
│   ├── const.py
│   ├── data.pt                   // Data used by model, containing training and testing data
│   ├── data_parse.py             // File for Processing data
│   ├── test_data.txt             // Original testing data
│   ├── test_process.txt          // Processed testing data
│   ├── train_data.txt            // Original training data
│   └── train_process.txt         // Processed training data
├── data_loder.py                 // Loading and processing data
├── main.py                       // Main function, containing training, testing and evaluation
├── model.py                      // BiLSTM model
└── model_bilstm                  // Trained model
##############################################

1. Preparing the data

If you want to change the data files, you can paste the data files in './data'.
Then change the filename in line 57 and 58 in data_parse.py.
Then execute Python command 'python data_parse.py'

2. Run the code

After preparing data, you can execute Python command 'python main.py' to train the model.
You can see the output in the console during the training process.



