import torch

class DeepLinearSvmWithRejector(torch.nn.Module):
    def __init__(self, features, dim_features:int, num_classes:int, init_weights=True):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of feature from body block.  
            num_classes: number of classification class.
        """
        super(DeepLinearSvmWithRejector, self).__init__()
        self.features = features
        self.dim_features = dim_features 
        self.num_classes = num_classes

        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # function r() in the original paper.
        # - r(x) >= 0: accept
        # - r(x)  < 0: reject
        # this implementation is following SelectiveNet [Geifman+, ICML2019].
        # to rescale (0,1) to (-1, 1). we replace sigmoid to tanh.
        self.rejector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Tanh()
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.rejector)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        prediction_out = self.classifier(x)
        rejection_out  = self.rejector(x)

        return prediction_out, rejection_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    import os
    import sys

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)

    from atro.vgg_variant import vgg16_variant

    features = vgg16_variant(32,0.3).cuda()
    model = DeepLinearSvmWithRejector(features,512,10).cuda()
    for m in model.classifier.modules():
        if isinstance(m, torch.nn.Linear):
            print(m.weight.shape)
            print(torch.matmul(m.weight.t(), m.weight).shape)
        