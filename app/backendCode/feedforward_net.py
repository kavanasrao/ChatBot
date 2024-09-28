
class NeuralNet(nn.Module):
    def __init__ (self, input_size,hidden_layer,num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size,hidden_layer)
        self.l2 = nn.Linear(hidden_layer,hidden_layer)
        self.l3 = nn.Linear(hidden_layer, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self,X):
        out = self.l1(X)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        
        return out 
