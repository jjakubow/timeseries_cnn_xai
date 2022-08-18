from torch import nn
import torch

class CNN(nn.Module):
    """This class implements a CNN with GradCAM"""
    def __init__(self, input_shape):
        super().__init__()
        
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        flat_size = self._infer_flat_size()
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(flat_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
        # placeholder for the gradients
        self.gradients = None
    
    def conv(self, x):
        # 1st convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 2nd convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 3rd convolution without pooling
        x = self.conv3(x)
        x = self.relu(x)
        
        return x
    
    def fully_connected(self, x):
        
        # flatten & fully connected layers
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        x = self.sigmoid(x)
        
        return x
    
    def forward(self, x, cam=False):
        # convolution
        x = self.conv(x)
        
        if cam:
            h = x.register_hook(self.activations_hook)
            
        x = self.pool(x)
        x = self.fully_connected(x)      

        return x
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.conv(x)
    
    def gradcam(self, x):
        """ Computes Grad-CAM """
        pred = self.forward(x, cam=True)
        pred.backward()
        gradients = self.get_activations_gradient()
        # print("Gradients shape: {}".format(gradients.shape))

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # print("Pooled Gradients shape: {}".format(pooled_gradients.shape))

        activations = self.get_activations(x).detach()
        # print("Activations shape: {}".format(activations.shape))

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        mean_activations = torch.mean(activations, dim=1).squeeze()
        mean_activations = torch.maximum(mean_activations, torch.tensor(0))
        mean_activations = self.reshape_transform(mean_activations, (x.shape[-2], x.shape[-1]))
        return mean_activations
      
    def _infer_flat_size(self):
        x = self.conv(torch.ones(*self.input_shape))
        x = self.pool(x)
        flat_size = x.view(-1).shape[0]
        
        return flat_size

    
    def reshape_transform(tensor, target_size):
        """ 
        Transforms a tensor to the required shape by interpolation
        Used to transform the tensor before last pooling layer to input size.
        Note: this should only interpolate time axis, not the feature axis!
        """
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, -1)
        tensor = tensor.reshape((1, 1, tensor.shape[0], tensor.shape[1]))
        image_with_single_row = tensor[:, None, :, :]
        # Lets make the time series into an image with 16 rows for easier visualization on screen later
        return torch.nn.functional.interpolate(tensor, target_size, mode='bilinear')


