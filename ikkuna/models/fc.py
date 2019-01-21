import torch


class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_shape, num_classes=1000, exporter=None):
        super(FullyConnectedModel, self).__init__()

        # if channel dim not present, add 1
        if len(input_shape) == 2:
            input_shape = list(input_shape) + [1]
        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            torch.nn.Linear(H * W * C, 4000),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(4000, 4000),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(4000, 4000),
            torch.nn.ReLU(inplace=False)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4000, 1000),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(1000, num_classes),
        )

        if exporter:
            exporter.set_model(self)
            exporter.add_modules(self)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        x = self.features(x)
        x = self.classifier(x)
        return x
