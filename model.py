import torchvision
import torch.nn as nn
import configuration as cfg
import torch

torch.hub.set_dir(cfg.MODEL_HUB)


def get_model(num_classes, mode="masks"):
    """
    Get the model based on the mode.
    TODO: improve the function to accept any backbone you want ?
    :args: num_classes: int, number of classes for our prb to adapt the model
    :args: mode: int, mode to choose which model to use (segmentation model, bounding boxes model, etc)
    return the model
    """
    if mode == "deeplab":
        # Not really well working
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    if mode == "masks":
        # 'Best one' right now
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    elif mode == "polygones":
        # Not really tried a lot with that, but working
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


def check_trainable_params(model, verbose=False):
    """Vérifie et affiche les paramètres entraînables"""
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if verbose:
                print(f"✓ {name}: {param.numel():,} params")
        else:
            if verbose:
                print(f"✗ {name}: {param.numel():,} params (frozen)")

    print(f"Params: {trainable_params:,}/{total_params:,} trainable ({100 * trainable_params / total_params:.1f}%)")

if __name__ == "__main__":
    num_classes = 1 + 1  # background + class, ie rooftop
    exec = "deeplab"

    model = get_model(num_classes, mode=exec).to(cfg.DEVICE)
    check_trainable_params(model)