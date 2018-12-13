import torch

class AdamModel(torch.nn.Module):

    def __init__(self, exporter, input_shape):

        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Conv2d(C, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(C, 64, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(C, 128, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # for each conv or padding layer, apply the output size formula ((length - filter_size +
        # 2*padding)/ stride) + 1
        H_conv1    = (H - 5 + 2 * 0) // 1 + 1
        H_mp1      = (H_conv1 - 2 * 0) // 2 + 1
        H_conv2    = (H_mp1 - 5 + 2 * 0) // 1 + 1
        H_mp2      = (H_conv2 - 2 * 0) // 2 + 1
        H_conv3    = (H_mp2 - 5 + 2 * 0) // 1 + 1
        H_mp3      = (H_conv3 - 2 + 2 * 0) // 2 + 1
        self.H_out = (H_mp3 - 2 + 2 * 0) // 2 + 1

        W_conv1    = (W - 5 + 2 * 0) // 1 + 1
        W_mp1      = (W_conv1 - 2 * 0) // 2 + 1
        W_conv2    = (W_mp1 - 5 + 2 * 0) // 1 + 1
        W_mp2      = (W_conv2 - 2 * 0) // 2 + 1
        W_conv3    = (W_mp2 - 5 + 2 * 0) // 1 + 1
        W_mp3      = (W_conv3 - 2 + 2 * 0) // 2 + 1
        self.W_out = (W_mp3 - 2 + 2 * 0) // 2 + 1

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.H_out * self.W_out * 128, 1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(1000, 10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x
