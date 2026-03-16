
import argparse
from PIL import Image
import torch
from torchvision import transforms
import torchvision
import os

def get_transform():
    """
    This should match the transform used in dataset.py
    Modify here if your training transform changes.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform


def convert_image(input_path, output_path):

    if not os.path.exists(input_path):
        print("Input image not found!")
        return

    transform = get_transform()

    # Load image
    image = Image.open(input_path).convert("RGB")

    # Apply same preprocessing as training
    tensor_image = transform(image)

    # Save the processed tensor as image
    torchvision.utils.save_image(tensor_image, output_path)

    print("Processed image saved to:", output_path)
    print("Tensor shape fed to model:", tensor_image.shape)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert masked image to model input format")
    parser.add_argument("--input", required=True, help="Path to masked input image")
    parser.add_argument("--output", default="processed_image.png", help="Output image path")

    args = parser.parse_args()

    convert_image(args.input, args.output)

