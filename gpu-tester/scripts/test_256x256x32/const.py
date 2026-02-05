"""Constants for 256x256x32 VRAM tests."""

SPATIAL = 256
DEPTH = 32
IN_CHANNELS = 3
NUM_CLASSES = 2  # for CrossEntropyLoss

BATCH_SIZES = [4, 8, 12, 16, 20, 24, 32, 48, 64]
DEFAULT_FAMILIES = ["ResNet3D", "SE-ResNet3D", "ViT3D", "ConvNeXt3D"]
