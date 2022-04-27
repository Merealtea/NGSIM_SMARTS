import torch
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,squence_len):
        super(PolicyNetwork,self).__init__()
        self.lstm = nn.LSTM(state_dim,state_dim,squence_len)
        self.fc1 = nn.Linear(state_dim*squence_len,200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200,50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50,10)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10,action_dim)
        self.squence_len = squence_len
        self.state_dim = state_dim
    
    def forward(self, input,c0,h0):
        output,(h,c) = self.lstm(input,(c0,h0))
        output = output.transpose(0,1)
        # TODO add the final_dim
        output = output.view(-1,self.squence_len * self.state_dim)

        output = self.fc1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        return output

class Discriminator(nn.Module):
    def __init__(self,state_dim,action_dim,squence_len):
        super(Discriminator,self).__init__()
        self.lstm = nn.LSTM(state_dim,state_dim,squence_len)
        self.fc1 = nn.Linear(state_dim*squence_len + action_dim,200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200,50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50,10)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
        self.squence_len = squence_len
        self.state_dim = state_dim
    
    def forward(self, input,c0,h0,action):
        pro,(h,c) = self.lstm(input,(c0,h0))
        pro = pro.transpose(0,1)
        # TODO add the final_dim
        pro = pro.view(-1,self.squence_len * self.state_dim)
        pro = torch.act([pro,action],dim = 1)
        pro = self.fc1(pro)
        pro = self.relu1(pro)
        pro = self.fc2(pro)
        pro = self.relu2(pro)
        pro = self.fc3(pro)
        pro  = self.relu3(pro)
        pro  = self.sigmoid(self.fc4(pro))
        return pro