import torch as th

class MLP(th.nn.Module):
    def __init__(self, input_size=44, hidden_sizes=[200, 120, 70], output_size=20):
        super(MLP, self).__init__()
        self.dense_layers = th.nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.dense_layers.append(th.nn.Linear(input_size, hidden_sizes[i]))
            elif i == len(hidden_sizes)-1:
                self.dense_layers.append(th.nn.Linear(hidden_sizes[i-1], output_size))
            else:
                self.dense_layers.append(th.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.initialize_weights()

    def forward(self, x):
        for i in range(len(self.dense_layers)-1):
            x = th.relu(self.dense_layers[i](x))
        x = self.dense_layers[-1](x)
        x = th.softmax(x, dim=-1)
        return x
    
    def initialize_weights(self):
        for layer in self.dense_layers:
            th.nn.init.xavier_uniform_(layer.weight)
            th.nn.init.zeros_(layer.bias)