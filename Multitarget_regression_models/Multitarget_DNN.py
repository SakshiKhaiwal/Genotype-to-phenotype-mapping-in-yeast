import numpy as np
import torch
from torch import Tensor
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


# Loading data

Predictor_matrix = pd.read_csv('path_to_predictor_matrix.csv')
Target_matrix = pd.read_csv('path_to_target_matrix.csv')
Metadata_matrix = pd.read_csv('path_to_metadata.csv')

Predictor_matrix = Predictor_matrix.fillna(Predictor_matrix.mean())
Target_matrix = Target_matrix.fillna(Target_matrix.mean())
scaler = StandardScaler()
Target_matrix = scaler.fit_transform(Target_matrix)
Predictor_matrix = Predictor_matrix.to_numpy()

# Convert Input and Output data to Tensors and create a TensorDataset

input = torch.tensor(Predictor_matrix)      # Create tensor of type torch.float32
print('\nInput format: ', input.shape, input.dtype)
output = torch.tensor(Target_matrix)        # Create tensor type torch.int64
print('Output format: ', output.shape, output.dtype)
data = TensorDataset(input, output)    # Create a torch.utils.data.TensorDataset object for further data manipulation

# Split to Train, Validate and Test sets using random_split
train_batch_size = 8
number_rows = len(input)
test_split = int(number_rows * 0.3)
validate_split = int(number_rows * 0.2)
train_split = number_rows - test_split - validate_split
train_set, validate_set, test_set = random_split(
    data, [train_split, validate_split, test_split])





# Create Dataloader to read the data within batch sizes and put into memory
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
validate_loader = DataLoader(validate_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

# Define model parameters
input_size = list(input.shape)[1]
learning_rate = 0.025
output_size = list(output.shape)[1]


# Define neural network

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(input_size, 7500)
        self.layer2 = nn.Linear(7500, 7000, nn.Dropout(0.5))
        self.layer2_ = nn.Linear(7000, 7000, nn.Dropout(0.5))
        self.layer3 = nn.Linear(7000, 6000, nn.Dropout(0.5))
        self.layer3_ = nn.Linear(6000, 6000, nn.Dropout(0.5))
        self.layer4 = nn.Linear(6000, 5000, nn.Dropout(0.5))
        self.layer4_ = nn.Linear(5000, 5000, nn.Dropout(0.5))
        self.layer5 = nn.Linear(5000, 2500, nn.Dropout(0.5))
        self.layer5_ = nn.Linear(2500, 2500, nn.Dropout(0.5))
        self.layer6 = nn.Linear(2500, 1000, nn.Dropout(0.5))
        self.layer6_ = nn.Linear(1000, 1000, nn.Dropout(0.5))
        self.layer7 = nn.Linear(1000, 500, nn.Dropout(0.5))
        self.layer8 = nn.Linear(500, 100, nn.Dropout(0.25))
        self.layer9 = nn.Linear(100, 500, nn.Dropout(0.25))
        self.layer10 = nn.Linear(500, 1000, nn.Dropout(0.25))
        self.layer10_ = nn.Linear(1000, 1000, nn.Dropout(0.5))
        self.layer11 = nn.Linear(1000, 2500, nn.Dropout(0.5))
        self.layer11_ = nn.Linear(2500, 2500, nn.Dropout(0.5))
        self.layer12 = nn.Linear(2500, 5000, nn.Dropout(0.5))
        self.layer12_ = nn.Linear(5000, 5000, nn.Dropout(0.5))
        self.layer13 = nn.Linear(5000, 6000, nn.Dropout(0.5))
        self.layer13_ = nn.Linear(6000, 6000, nn.Dropout(0.5))
        self.layer14 = nn.Linear(6000, 7000, nn.Dropout(0.5))
        self.layer14_ = nn.Linear(7000, 7000, nn.Dropout(0.5))
        self.layer15 = nn.Linear(7000, 7500, nn.Dropout(0.5))
        self.layer16 = nn.Linear(7500, input_size, nn.Dropout(0.5))
        self.layer17 = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor):
        residual = x
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x2_ = F.relu(self.layer2_(x2))
        x3 = F.relu(self.layer3(x2_))
        x3_ = F.relu(self.layer3_(x3))
        x4 = F.relu(self.layer4(x3_))
        x4_ = F.relu(self.layer4_(x4))
        x5 = F.relu(self.layer5(x4_))
        x5_ = F.relu(self.layer5_(x5))
        x6 = F.relu(self.layer6(x5_))
        x6_ = F.relu(self.layer6_(x6))
        x7 = F.relu(self.layer7(x6_))
        x8 = F.relu(self.layer8(x7))
        x9 = F.relu(self.layer9(x8))
        x10 = F.relu(self.layer10(x9))
        x10_ = F.relu(self.layer10_(x10))
        x11 = F.relu(self.layer11(x10_+x6_))
        x11_ = F.relu(self.layer11_(x11))
        x12 = F.relu(self.layer12(x11_ + x5_))
        x12_ = F.relu(self.layer12_(x12))
        x13 = F.relu(self.layer13(x12_+x4_))
        x13_ = F.relu(self.layer13_(x13))
        x14 = F.relu(self.layer14(x13_+x3_))
        x14_ = F.relu(self.layer14_(x14))
        x15 = F.relu(self.layer15(x14_+x2_))
        x16 = F.relu(self.layer16(x1+x15))
        x16 = x16 + residual
        return self.layer17(x16)


model = Network(input_size, output_size)

# Define your execution device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device\n")
model.to(device)

# Function to save the model
def saveModel():
    path = "./NN_regmat_PA.pth"
    torch.save(model.state_dict(), path)


# Define the loss function with MSE loss and an optimizer with Adam optimizer
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.1)


# Training Function
def train(num_epochs):
    best_accuracy = 0.0
    val_epoch = 5
    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        # Training Loop
        for data in train_loader:
            # for data in enumerate(train_loader, 0):
            inputs, outputs = data  # get the input and real species as outputs; data is a list of [inputs, outputs]
            inputs = inputs.float()
            outputs = outputs.float()
            optimizer.zero_grad()  # zero the parameter gradients
            torch.autograd.set_detect_anomaly(True)
            predicted_outputs = model(inputs)  # predict output from the model
            train_loss = loss_fn(predicted_outputs, outputs)  # calculate loss for the predicted output

            train_loss.backward()  # backpropagate the loss

            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value
        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)
        if (epoch % val_epoch == 0) and (epoch > 0):
            # Validation Loop
            with torch.no_grad():
                model.eval()
                for data in validate_loader:
                    inputs, outputs = data
                    inputs = inputs.float()
                    outputs = outputs.float()
                    predicted_outputs = model(inputs)
                    val_loss = loss_fn(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                    running_vall_loss += val_loss.item()
                    total += outputs.size(0)
                    running_accuracy += (pearsonr(outputs[0].cpu().detach().numpy(), predicted_outputs[0].cpu().detach().numpy())[0])

                # Calculate validation loss value
            val_loss_value = running_vall_loss / len(validate_loader)

            # Calculate accuracy as the number of average Pearson's coefficient in the validation batch divided by the total number of predictions done.
            accuracy = (100 * running_accuracy / total)

            # Save the model if the accuracy is the best
            if accuracy > best_accuracy:
                saveModel()
                best_accuracy = accuracy

            # Print the statistics of the epoch
            print('Completed training batch', epoch, 'Training Loss is: %4f' % train_loss_value,
                  'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is %d %%' % accuracy)

    return predicted_outputs, outputs, running_accuracy


# Function to test the model
def test():
    # Load the model that we saved at the end of the training loop
    model = Network(input_size, output_size)
    path = "./NN_regmat_PA.pth"
    model.load_state_dict(torch.load(path))
    predicted_output_matrix = []
    output_matrix = []
    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            inputs = inputs.float()
            outputs = outputs.float()
            output_matrix.append(outputs)
            predicted_outputs = model(inputs)
            predicted_output_matrix.append(predicted_outputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += pearsonr(outputs[0].cpu().detach().numpy(),predicted_outputs[0].cpu().detach().numpy())[0]

        print('Accuracy of the model based on the test set of', test_split,
              'inputs is: %d %%' % (100 * running_accuracy / total))

    return output_matrix, predicted_output_matrix





if __name__ == "__main__":
    num_epochs = 5
    train_output = train(num_epochs)
    print('Finished Training\n')
    test_output = test()


for k in range(0, len(test_output[0])):
    test_output[0][k] = test_output[0][k].tolist()[0]
    test_output[1][k] = test_output[1][k].tolist()[0]


test_matrix = np.reshape(test_output[0], (test_split, output_size)).T
test_matrix_predicted = np.reshape(test_output[1], (test_split, output_size)).T

test_set_metadata = Metadata_matrix.iloc[test_set.indices, :]
test_set_metadata.to_csv('Test_set_metadata_PA.csv')
pd.DataFrame(test_matrix).to_csv('Test_matrix_true_using_PA.csv')
pd.DataFrame(test_matrix_predicted).to_csv('Test_matrix_predicted_using_PA.csv')

