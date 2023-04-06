import torch

class AlexNet():
    def Alex_model():
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model.eval()
        return model