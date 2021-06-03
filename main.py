import io
import os
import json
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torchvision.models as models
from torchvision import datasets, transforms
import uuid
import os
device = 'cpu'
model = models.densenet121(pretrained=True)

st.write('Welcome to NeuralStyle app')

image_style = None
image_content = None

image_content = st.file_uploader("Choose the Content Image")

image_style = st.file_uploader("Choose the Style Image")

uniq_name = str(uuid.uuid4())

# parser = argparse.ArgumentParser()
# parser.add_argument('--max_size', type=int, default=400)
# parser.add_argument('--total_step', type=int, default=2000)
# parser.add_argument('--log_step', type=int, default=10)
# parser.add_argument('--sample_step', type=int, default=500)
# parser.add_argument('--style_weight', type=float, default=100)
# parser.add_argument('--lr', type=float, default=0.003)


total_step = st.sidebar.selectbox(
    'total steps', [10, 50, 100, 300, 500, 1000, 1500, 2000], index=3)

max_size = 400


style_weight = st.sidebar.slider("Style Weight", 100, 500, value=200)

sample_step = st.sidebar.selectbox('Sample Steps after which to display output', [
                                   5, 10, 50, 100, 500], index=2)

log_step = st.sidebar.selectbox(
    'Steps after which to display progress', [1, 2, 5, 10, 25, 50], index=1)

lr = st.sidebar.selectbox(
    'Learning Rate', [0.001, 0.003, 0.005, 0.01], index=1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])

left_column, right_column = st.beta_columns(2)

with left_column:
    if image_content is None:
        st.write("No Content Image Provided.")
    else:
        st.markdown("**Content Image**")
        content = Image.open(image_content)
        content.thumbnail((300, 300))
        st.image(content)
        content_image = content

with right_column:
    if image_style is None:
        st.write("No Style Image Provided.")
    else:
        st.write("**Style Image**")
        style = Image.open(image_style)
        style.thumbnail((300, 300))
        st.image(style)
        style_image = style


def load_image(image, transform=None, max_size=None, shape=None):
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    if shape:
        image = image.resize(shape, Image.LANCZOS)
    if transform:
        image = transform(image).unsqueeze(0)
    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


m = 0


if st.button("Run"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

    content = load_image(content_image, transform, max_size=max_size)
    style = load_image(style_image, transform, shape=[
                       content.size(2), content.size(3)])

    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=lr, betas=[0.5, 0.999])
    vgg = VGGNet().to(device).eval()

    for step in range(total_step):
        m = m + 1
        st.spinner("In Progress...")

        # Extract multiple(5) conv feature vectors
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):

            content_loss += torch.mean((f1 - f2)**2)

            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3)**2) / (c * h * w)

        loss = content_loss + style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % log_step == 0:
            st.write('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                     .format(step + 1, total_step, content_loss.item(), style_loss.item()))

        if (step + 1) % sample_step == 0:
            # Save the generated image
            denorm = transforms.Normalize(
                (-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            name = 'output-{} steps-'.format(step + 1) + uniq_name + '.png'
            torchvision.utils.save_image(img, name)
            st.image(Image.open(name),
                     "Image after {} steps.".format(step + 1))
            os.remove(name)
else:
    st.write("Press Run to Execute.")


if (m == total_step):
    st.success("Done!")
