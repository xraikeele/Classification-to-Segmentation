import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model  # For models from the `timm` library

def load_resnet(weight, num_classes=1):
    """
    Loads a pre-trained ResNet model and modifies its classification head for binary classification.
    Args:
        weight (str): Name of pretrained weights to load.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: ResNet model.
    """
    model = models.resnet50(weights=weight)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_efficientnet(weight, num_classes=1):
    """
    Loads a pre-trained EfficientNet model and modifies its classification head for binary classification.
    Args:
        weight (str): Name of pretrained weights to load.
        num_classes (int): Number of output classes.
        model_name (str): Name of the EfficientNet model (e.g., 'efficientnet_b0').
    Returns:
        nn.Module: EfficientNet model.
    """
    model = models.efficientnet_b0(weights=weight)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_mobilenet(weight, num_classes=1):
    """
    Loads a pre-trained MobileNet model and modifies its classification head for binary classification.
    Args:
        weight (str): Name of pretrained weights to load.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: MobileNetV2 model.
    """
    model = models.mobilenet_v2(weights=weight)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def load_vit(weight, num_classes=1):
    """
    Loads a pre-trained ViT model and modifies its classification head for binary classification.
    Args:
        weights (str): Whether to load pretrained weights.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: ViT model.
    """
    model = models.vit_b_16(weights=weight)
    # Get the number of input features to the classification head
    in_features = model.heads.head.in_features
    
    # Replace the classification head for binary classification
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

def load_swin(weight, num_classes=1):
    """
    Loads a pre-trained swin model and modifies its classification head for binary classification.
    Args:
        weights (str): Whether to load pretrained weights.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: swin model.
    """
    model = models.swin_s(weights=weight)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model

def load_model(model_name, num_classes=1):
    """
    General function to load a model by name.
    Args:
        model_name (str): Name of the model to load (e.g., 'resnet50', 'mobilenet_v2', 'efficientnet_b0').
        pretrained (bool): Whether to load pretrained weights.
        num_classes (int): Number of output classes.
    Returns:
        nn.Module: Pre-trained model with modified classification head.
    """
    if model_name == "resnet50":
        return load_resnet('IMAGENET1K_V2',num_classes)
    elif model_name == "mobilenet_v2":
        return load_mobilenet('IMAGENET1K_V2',num_classes)
    elif model_name == "efficientnet_b0":
        return load_efficientnet('IMAGENET1K_V1',num_classes)
    elif model_name == "ViT":
        return load_vit('IMAGENET1K_V1',num_classes)
    elif model_name == "swin":
        return load_swin('IMAGENET1K_V1', num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

if __name__ == "__main__":
    # Example usage
    model_name = "swin"
    model = load_model(model_name, num_classes=1)
    print(f"Loaded model: {model_name}")
    print(model)