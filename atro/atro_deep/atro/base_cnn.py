import torch

class BaseCNN(torch.nn.Module):
    def __init__(self, input_size:int, dropout_base_prob:float=0.5, init_weights=True):
        """
        Args:
            input_size: size of input.
            dropout_base_prob: base probability of an element to be zeroed by dropout.
            init_weights: initialize weight or not.
        """

        super(BaseCNN, self).__init__()
        self.dropout_base_prob = dropout_base_prob
        self.feature_size = input_size-10 # conv x2 + pool x2 

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(int(self.feature_size**2)*64, 1024),
            torch.nn.Dropout(self.dropout_base_prob),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = BaseCNN(input_size=32).cuda()

    x = torch.randn(16,3,32,32).cuda()
    out = model(x)
    print(out.shape)

