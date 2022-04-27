import torch
from torch import nn
from torch.distributions.normal import Normal

class Policynet(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(Policynet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.activate1 = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm1d(100)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(100, 500)
        self.activate2 = nn.LeakyReLU(0.1)
        self.bn2 = nn.BatchNorm1d(500)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(500,100)
        self.activate3 = nn.LeakyReLU(0.1)
        self.bn3 = nn.BatchNorm1d(100)
        self.drop3 = nn.Dropout(0.2)

        self.mean = nn.Linear(100,action_dim)
        self.activate4 = nn.Tanh()
        self.log_std = nn.Linear(100,action_dim)
        
        self.fc1.weight.data.uniform_(-0.001, 0.001)
        self.fc2.weight.data.uniform_(-0.001, 0.001)
        self.fc3.weight.data.uniform_(-0.001, 0.001)

        self.mean.weight.data.uniform_(-0.001, 0.001)
        self.log_std.weight.data.uniform_(-0.001, 0.001)

        self.action_dim = action_dim

    def forward(self, state,tetminatic = False):
        out = self.fc1(state)
        out = self.activate1(out)
        out = self.drop1(self.bn1(out))

        out = self.fc2(out)
        out = self.activate2(out)
        out = self.drop2(self.bn2(out))

        out = self.fc3(out)
        out = self.activate3(out)
        out =  self.drop3(self.bn3(out))

        mean = self.activate4(self.mean(out))
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        distb = Normal(mean, std)
        if tetminatic:
            return mean
        return distb,std,mean

class Valuenet(nn.Module):
    def __init__(self,state_dim):
        super(Valuenet,self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.activate1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(200)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(200, 500)
        self.activate2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(500)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(500,200)
        self.activate3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(200)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(200,1)
    
    def cal_value(self,state):
        out = self.fc1(state)
        out = self.activate1(out)
        out = self.drop1(self.bn1(out))

        out = self.fc2(out)
        out = self.activate2(out)
        out = self.drop2(self.bn2(out))

        out = self.fc3(out)
        out = self.activate3(out)
        out =  self.drop3(self.bn3(out))

        out = self.fc4(out)
        return out

class Discriminator(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Discriminator,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 200)
        self.activate1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(200)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(200, 200)
        self.activate2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(200)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(200,1)
        self.sigmoid = nn.Sigmoid()

        self.fc1.weight.data.uniform_(-0.001, 0.001)
        self.fc2.weight.data.uniform_(-0.001, 0.001)
        self.fc3.weight.data.uniform_(-0.001, 0.001)

    def cal_pro(self,state,action):
        input = torch.cat([state,action],dim = 1)
        out = self.fc1(input)
        out = self.activate1(out)
        out = self.drop1(self.bn1(out))

        out = self.fc2(out)
        out = self.activate2(out)
        out = self.drop2(self.bn2(out))

        out = self.sigmoid(self.fc3(out))
        return out

class Qnet(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 500)
        self.activate1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(500)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(500, 500)
        self.activate2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(500)
        self.drop2 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(500,1)

        self.fc1.weight.data.uniform_(-0.001, 0.001)
        self.fc2.weight.data.uniform_(-0.001, 0.001)
        self.fc4.weight.data.uniform_(-0.001, 0.001)

    def cal_q_val(self,state,act):
        input = torch.cat([state,act],dim = 1)
        out = self.fc1(input)
        out = self.activate1(out)
        out = self.drop1(self.bn1(out))

        out = self.fc2(out)
        out = self.activate2(out)
        out = self.drop2(self.bn2(out))

        out = self.fc4(out)
        return out


