import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """

        self.x = x
        self.z1 = self.parameters["W1"] @ self.x.t() + self.parameters["b1"].view(-1,1)

        self.z2 = self.f_function(self.z1)
        self.z3 = self.parameters["W2"] @ self.z2 + self.parameters["b2"].view(-1,1)
        print(self.parameters["b2"].view(-1,1))
        self.y_pred = self.g_function(self.z3)
        return self.y_pred
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        dy_hatdz3 = -torch.exp(self.z3)/pow(1+torch.exp(self.z3), 2)
        dy_hatdz3 = torch.diag(dy_hatdz3.view(-1))

        dz2dz1 = (torch.sign(self.z1) + 1)/2
        dz2dz1 = torch.diag(dz2dz1.view(-1))
        for i in range()
        dJdwij = dJdy_hat @ dy_hatdz3 @ self.parameters["W2"] @ dz2dz1 @ x[0][0]
        self.grads["dJdW1"] = dJdy_hat @ dy_hatdz3 @ self.parameters["W2"] @ dz2dz1 #@ self.x.t()
        print(dz2dz1.size())

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    pass


    # return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    pass

    # return loss, dJdy_hat











