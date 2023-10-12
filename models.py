import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class LinearQNet(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)


    def forward(self, x):
        # model: in -> linear -> relu -> linear -> out
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    

    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


        
class QNetTrainer:
    def __init__(self, model: LinearQNet, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.lr)
        self.cretirion = nn.MSELoss()


    def train(self, states, actions, next_states, rewards, dones):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = (dones, )

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            next_states = torch.unsqueeze(next_states, 0)
            rewards = torch.unsqueeze(rewards, 0)

        # compute Q-value
        prediction = self.model(states)

        # compute target Q-value 
        target = prediction.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][torch.argmax(actions[idx]).item()] = Q_new

        
        self.optimizer.zero_grad()
        # compute loss
        loss = self.cretirion(target, prediction)

        # compute gradient
        loss.backward()

        # update parameter
        self.optimizer.step()

        


        