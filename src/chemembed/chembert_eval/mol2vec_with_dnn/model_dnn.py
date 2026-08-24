import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_Class(nn.Module):
    def __init__(self, input_dim=420, output_dim=300):
        super(DNN_Class, self).__init__()

        self.fc1 = nn.Linear(input_dim, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(1024, 300)  # Number of classes

        self.dropout = nn.Dropout(0.25)

    def encode(self, x):
        # Flatten input if it comes with extra dimensions (e.g., (B, 1, 420))
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)  # Final feature layer
        return x

    def forward(self, x):
        x = self.encode(x)
        return x
