"""
The file can be used to train a RNN based model to predict sentiment of the given text.

To execute the file, run: python main.py --model=RNN
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Imports - our files
import utils.helper_functions as utils
import utils.models as models
import argparse

# Global definitions - data
DATA_FN = 'data/crowdflower_emotion.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128  #128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image 
learning_rate = 0.001
num_epochs = 200
hidden1_dim = 100 # number of hidden layers for RNN network
learning_decay_rate = 0.4

# Global definitions - saving and loading data
FRESH_START = True  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.


def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of your models, to be trained to perform 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    all_train_losses = []
    all_train_accuracy = []
    all_val_losses = []
    all_val_accuracy = []
    # learning_rates = [0.0005, 0.0001, 0.00005]
    temp = 0
    temp_2=0

    #early stopping
    patience = 4
    wait = 0

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        
        model.train()
        for i, (X, y) in enumerate(train_generator):
            #zero the gradients
            optimizer.zero_grad()

            predictions = model(X)
            
            loss = loss_fn(predictions, y)

            train_acc = get_accuracy(predictions, y.view(-1), BATCH_SIZE)

            # loss = loss_fn(torch.tensor(outputs), y)
            loss.backward()
            
            optimizer.step()

            epoch_train_loss += loss.detach().item()
            # print("\n\nTraining loss for batch {0} is: {1}".format(i, loss.detach().item()))
            
            # print("Training accuracy for batch {0}: {1}".format(i, train_acc))
            epoch_train_acc += train_acc


        # model.eval()
        print("[EPOCH]: {0} end".format(epoch))
        all_train_accuracy.append(epoch_train_acc/i)
        all_train_losses.append(epoch_train_loss/i)
        print("Training accuracy: {0}".format(epoch_train_acc/i))
        print("Training loss: {0}".format(epoch_train_loss/i))

        if epoch % 10 ==0 and epoch > 0:
            # if temp < len(learning_rates):
            for g in optimizer.param_groups:
                print("[Learning Rate]: before {0}".format(g['lr']))
                g['lr'] = g['lr']*learning_decay_rate
                print("[Learning Rate]: after {0}".format(g['lr']))

        model.eval()
        for i, (X_val, y_val) in enumerate(dev_generator):
            val_predictions = model(X_val)

            val_loss = loss_fn(val_predictions, y_val)
            epoch_val_loss += val_loss.detach().item()

            epoch_val_acc += get_accuracy(val_predictions, y_val.view(-1), BATCH_SIZE)

        all_val_accuracy.append(epoch_val_acc/i)
        all_val_losses.append(epoch_val_loss/i)
        print("Validation accuracy: {0}".format(epoch_val_acc/i))
        print("Validation loss: {0}".format(epoch_val_loss/i))

        # Early stopping
        if epoch > 5:
            if all_val_losses[-1] > all_val_losses[-2]:
                wait += 1
                if wait > patience:
                    break
            else:
                wait = 0

        # saving models
        # if epoch > 20 and epoch % 5 == 0:
        #     torch.save(model, "data/models/model_saved_ep{}_.pkl".format(epoch))


    # print("All accuracy:")
    # print(all_train_accuracy)
    # print("All losses:")
    # print(all_train_losses)
    with open("data/training_acc_loss.txt", 'a') as log_file:
        log_file.write("\n[Training Accuracy for epoch{0}_lr{1}_batch{2}_hidDim{3}]:\n".format(num_epochs, 
            learning_rate, BATCH_SIZE, hidden1_dim))
        log_file.write(",".join(str(acc) for acc in all_train_accuracy))
        log_file.write("\n[Training Losses]:\n")
        log_file.write(",".join(str(lss) for lss in all_train_losses))

        log_file.write("\n\n[Validation Accuracy]:\n")
        log_file.write(",".join(str(acc) for acc in all_val_accuracy))
        log_file.write("\n[Validation Losses]:\n")
        log_file.write(",".join(str(lss) for lss in all_val_losses))

    torch.save(model.state_dict(), "model.pkl")
    # torch.save(model, "data/models/model_saved_ep{0}_lr{1}_bt{2}_hid{3}.pkl".format(num_epochs, 
    #     learning_rate, BATCH_SIZE, hidden1_dim))

    return model


    # raise NotImplementedError


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)
            # print("X_b dimentions: {0}".format(X_b.shape))
            # print("y_pred dimentions: {0}".format(y_pred.shape))
            # print("y_b dimentions: {0}".format(y_b.shape))

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))
    with open("data/test_acc_loss.txt", 'a') as log_test_file:
        log_test_file.write("[Testing Accuracy for epoch{0}_lr{1}_batch{2}_hidDim{3}]:\n".format(num_epochs, 
            learning_rate, BATCH_SIZE, hidden1_dim))
        log_test_file.write("Test Loss: {}\n".format(loss))
        log_test_file.write("Test F-score: {}\n".format(f1_score(gold, predicted, average='macro')))


def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    DATA_FN = args.inputfile
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields if you would like to use them at any time
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                BATCH_SIZE,
                                                                                                EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")
            

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    loss_fn.requires_grad = True
    print(USE_CUDA)

    input_size = len(train_data.word2idx)
    # print("Input size is: {}".format(len(train_data.word2idx)))
    
    print("The embedding glove matrix size: {}".format(embeddings.shape))

    if args.model == "RNN":
        model = models.RecurrentNetwork(embeddings, num_class=NUM_CLASSES)
        # model = models.RecurrentNetwork(input_size, EMBEDDING_DIM, hidden1_dim, output_size=NUM_CLASSES)
    elif args.model == "LSTM":
        model = models.LSTM(input_size, EMBEDDING_DIM, hidden1_dim, output_size=NUM_CLASSES)
    # elif args.model == "GRU":
    #     model = models.RecurrentNetwork(input_size, EMBEDDING_DIM, hidden1_dim, output_size=NUM_CLASSES)
    # assigning the glove embedding values to the embedding layer
    # model.embedding.weight.data = embeddings
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    for param in model.parameters():
        param.requires_grad = True
    # print("Total training examples: {0}".format(len(train_generator)*BATCH_SIZE))
    # print("Model rnn weight snapshot")
    # print(model.fc.weight[:10,:10])

    model_trained = train_model(model, loss_fn, optimizer, train_generator, dev_generator) #loss_fn, optimizer, train_generator, dev_generator
    # print(model.in2hidden1.weight[1:10, 1:20])
    # print("Model rnn weight snapshot")
    # print(model_trained.fc.weight[:10,:10])  
    # model_trained = torch.load("data/models/model_saved_ep15_lr0.001_bt128_hid100_best_model.pkl")  
    test_model(model_trained, loss_fn, test_generator) #model, loss_fn, test_generator
    # raise NotImplementedError


def get_accuracy(y_pred, y_target, batch_size):
    corrects = (torch.max(y_pred, 1)[1].view(y_target.size()).data == y_target.data).sum()
    accuracy = 100.0*corrects/y_target.size()[0]
    return accuracy.item()


def hyper_param_search(args, learning_rates, hidden_size, batch_sizes, epoch_list):
    global learning_rate
    global hidden1_dim
    global BATCH_SIZE
    global num_epochs
    for learning_rate_ in learning_rates:
        learning_rate = learning_rate_
        for BATCH_SIZE_ in batch_sizes:
            BATCH_SIZE = BATCH_SIZE_
            for num_epochs_ in epoch_list:
                num_epochs = num_epochs_
                for hidden1_dim_ in hidden_size:
                    hidden1_dim = hidden1_dim_
                    main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["RNN", "LSTM", "GRU"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--inputfile', dest='inputfile', required=False,
                        default='data/crowdflower_emotion.csv',
                        help='The name of the input csv file with sentiment as first column and text in the second column')
    args = parser.parse_args()
    main(args)
    #hyperparamter tuning
    # learning_rates = [0.01, 0.001, 0.0005, 0.0001]
    # hidden_size = [50, 100, 150]
    # batch_sizes = [64, 90, 128]
    # epoch_list = [40, 60, 80, 100, 120, 140, 160]
    # hyper_param_search(args, learning_rates, hidden_size, batch_sizes, epoch_list)
