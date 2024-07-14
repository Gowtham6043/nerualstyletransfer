import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

    def load_image(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 244
    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    model = VGG().to(device).eval()
    model.device = device
    model.loader = loader

    original_image = model.load_image('gowtham.jpg.jpg')
    style_image = model.load_image('IMG20230711141750.jpg')

    generated = torch.randn(original_image.size()).to(device)
    optimizer = optim.LBFGS([generated])

    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01

    for step in range(total_steps):
        def closure():
            optimizer.zero_grad()
            generated_features = model(generated)
            original_features = model(original_image)
            style_features = model(style_image)
            
            style_loss = original_loss = 0

            for gen_features, orig_features, style_features in zip(generated_features, original_features, style_features):
                batch_size, channel, height, width = gen_features.shape

                original_loss += torch.mean((gen_features - orig_features)**2)
                
                G = gen_features.view(channel, height * width).mm(
                    gen_features.view(channel, height * width).t()
                )
                A = style_features.view(channel, height * width).mm(
                    style_features.view(channel, height * width).t()
                )
                style_loss += torch.mean((G - A)**2)

            total_loss = alpha * original_loss + beta * style_loss
            optimizer.zero_grad()
            total_loss.backward()
            return total_loss
        
        optimizer.step(closure)

        if step % 200 == 0:
            print(f"Step: {step}, Total Loss: {closure().item()}")
            save_image(generated, 'generated.png')

    # View the generated image
    generated_image = Image.open('generated.png')
    generated_image.show()