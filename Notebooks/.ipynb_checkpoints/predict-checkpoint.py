import argparse
import urllib.request

import torch
from torchvision import transforms, models
from PIL import Image
from safetensors.torch import load_file


def load_classes(filename="classes.txt"):
    """
    Loads class labels from a text file. Each line in the file is expected
    to have a class label with optional surrounding quotes and a trailing comma.
    
    Args:
        filename (str): Path to the classes text file.
    
    Returns:
        list[str]: A list of class labels.
    """
    classes = []
    with open(filename, "r") as f:
        for line in f:
            # Remove leading/trailing whitespace and any trailing commas.
            cleaned_line = line.strip().strip(",")
            # Remove surrounding quotes if present.
            cleaned_line = cleaned_line.strip("'").strip('"')
            if cleaned_line:
                classes.append(cleaned_line)
    return classes




def load_model(model_path: str) -> torch.nn.Module:
    """
    Load the EfficientNet_B4 model architecture and update its state with weights
    from the given safetensor file.
    
    Args:
        model_path (str): Path to the safetensor file.
        
    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    # Instantiate the EfficientNet_B4 model.
    # We assume the model has 1000 output classes (as in ImageNet1K)
    model = models.efficientnet_b4(num_classes=1000)
    
    # Load the state dict from the safetensor file.
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(args):
    # Load and transform the image
    image = Image.open(args.m).convert("RGB")
    # Using the provided transformer:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Load the model (update the path below if your safetensor is located elsewhere)
    model = load_model("../Models/Pokemon-efficientnet_b4-IMAGENET1K_V1-v1.safetensors")
    model.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Determine number of top predictions to show (default 5)
    top_n = args.n if args.n is not None else 5
    top_probs, top_indices = torch.topk(probabilities, top_n)

    # Load ImageNet class labels
    classes = load_classes()

    # Print top predictions
    print("Top predictions:")
    for i in range(top_n):
        label = classes[top_indices[i]]
        prob = top_probs[i].item() * 100
        print(f"{i + 1}. {label}: {prob:.2f}%")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict image class using Pokemon-efficientnet_b4 safetensor model."
    )
    parser.add_argument(
        "-m",
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)."
    )
    args = parser.parse_args()
    main(args)
