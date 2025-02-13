from torch import nn



class FTLL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Identify the last layer dynamically
        last_layer = None
        for name, module in self.model.named_children():
            last_layer = module  # Keep updating to get the last module
        
        # If a last layer exists, unfreeze its parameters
        if last_layer is not None:
            for param in last_layer.parameters():
                param.requires_grad = True
                
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    

class FTAL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    


class RTLL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Identify the last layer dynamically
        last_layer = None
        for name, module in self.model.named_children():
            last_layer = module  # Keep updating to get the last module

        # If a last layer exists, unfreeze its parameters
        if last_layer is not None:
            for param in last_layer.parameters():
                if param.dim() > 1:  # Reinitialize weight matrices, not biases
                    nn.init.xavier_uniform_(param)
                param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



class RTAL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Identify the last layer dynamically
        last_layer = None
        for name, module in self.model.named_children():
            last_layer = module  # Keep updating to get the last module

        # If a last layer exists, unfreeze its parameters
        if last_layer is not None:
            for param in last_layer.parameters():
                if param.dim() > 1:  # Reinitialize weight matrices, not biases
                    nn.init.xavier_uniform_(param)
                param.requires_grad = True

        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)