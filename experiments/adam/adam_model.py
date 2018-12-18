import torch


class AdamModel(torch.nn.Module):

    def __init__(self, input_shape, num_classes=1000, exporter=None):

        super().__init__()

        # if channel dim not present, add 1
        if len(input_shape) == 2:
            input_shape.append(1)
        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            # torch.nn.Dropout(),
            torch.nn.Conv2d(C, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # for each conv or padding layer, apply the output size formula ((length - filter_size + 2*padding)/ stride) + 1
        H_conv1 = ((H - 5 + 2*2) / 1) + 1
        H_mp1 = ((H_conv1 - 3 + 2*1) / 2) + 1
        H_conv2 = ((H_mp1 - 5 + 2*2) / 1) + 1
        H_mp2 = ((H_conv2 - 3 + 2*1) / 2) + 1
        H_conv3 = ((H_mp2 - 5 + 2*2) / 1) + 1
        H_mp3 = ((H_conv3 - 3 + 2*1) / 2) + 1

        W_conv1 = ((W - 5 + 2*2) / 1) + 1
        W_mp1 = ((W_conv1 - 3 + 2*1) / 2) + 1
        W_conv2 = ((W_mp1 - 5 + 2*2) / 1) + 1
        W_mp2 = ((W_conv2 - 3 + 2*1) / 2) + 1
        W_conv3 = ((W_mp2 - 5 + 2*2) / 1) + 1
        W_mp3 = ((W_conv3 - 3 + 2*1) / 2) + 1

        self.H_out = int(H_mp3)
        self.W_out = int(W_mp3)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.H_out * self.W_out * 128, 1000),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(),
            torch.nn.Linear(1000, num_classes),
            torch.nn.Softmax()
        )

        if exporter:
            exporter.set_model(self)
            exporter.add_modules(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x
