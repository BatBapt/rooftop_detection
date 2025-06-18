import os
import torch
import numpy as np
from torchvision.transforms import v2 as T
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import utils  # from wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
import yaml
import matplotlib.pyplot as plt
import model as my_model
import dataset as my_dataset
import configuration as cfg
import data_preprocessing as my_data_processing


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def run_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config


def setup_training(model, config, stage):
    config_stage = config[stage]

    if stage == "stage1":  # works with most all backbones
        # Freezer tout le backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Affirm heads are unfrozen
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        for param in model.rpn.parameters():
            param.requires_grad = True

    elif stage == "stage2":
        for param in model.backbone.parameters():
            param.requires_grad = True
        if hasattr(model.backbone, 'body'):
            for i, param in enumerate(model.backbone.body.parameters()):
                if i < 15:
                    param.requires_grad = False
        else:
            for i, layers in enumerate(model.backbone):
                if i < 10:
                    for param in layers.parameters():
                        param.requires_grad = False

    elif stage == "stage3":
        # Unfroze everything
        for param in model.parameters():
            param.requires_grad = True

    return config_stage


writer = SummaryWriter(log_dir='runs/rooftop')


def train_one_epoch(model, data_loader, device, optimizer, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Training Epoch {epoch}:"
    model.to(device)

    with tqdm(data_loader, desc=header) as tq:
        for i, (images, targets) in enumerate(tq):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            optimizer.zero_grad()
            losses_reduced.backward()
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Update tqdm postfix to display loss on the progress bar
            tq.set_postfix(loss=losses.item(), lr=optimizer.param_groups[0]["lr"])

            # Log losses to TensorBoard
            writer.add_scalar('Loss/train', losses.item(), epoch * len(data_loader) + i)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Loss/train_{k}', v.item(), epoch * len(data_loader) + i)

    print(f"Average Loss: {metric_logger.meters['loss'].global_avg:.4f}")
    writer.add_scalar('Loss/avg_train', metric_logger.meters['loss'].global_avg, epoch)
    return metric_logger.meters['loss'].global_avg


def evaluate(model, data_loader, device, epoch):
    model.eval()
    metric = MeanAveragePrecision()
    header = "Validation:"
    total_steps = len(data_loader)

    with torch.no_grad(), tqdm(total=total_steps, desc=header) as progress_bar:
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # Convert outputs for torchmetrics
            try:
                preds = [
                    {"masks": out["masks"], "boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
                    for out in outputs
                ]
                targs = [
                    {"masks": tgt["masks"], "labels": tgt["labels"], "boxes": tgt["boxes"]}
                    for tgt in targets
                ]
            except KeyError:  # masks key not present if we use bounding boxes
                preds = [
                    {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
                    for out in outputs
                ]
                targs = [
                    {"labels": tgt["labels"], "boxes": tgt["boxes"]}
                    for tgt in targets
                ]

            # Update metric for mAP calculation
            metric.update(preds, targs)
            progress_bar.update(1)

    results = metric.compute()

    # Log mAP to TensorBoard
    for k, v in results.items():
        if v.numel() == 1:  # Single element tensor
            writer.add_scalar(f'mAP/{k}', v.item(), epoch)
        else:  # Multi-element tensor, log each element separately
            for idx, value in enumerate(v):
                writer.add_scalar(f'mAP/{k}_{idx}', value.item(), epoch)
    return results


def progressive_training(model, train_loader, val_loader, device, save_dir):
    """Lance l'entraînement progressif complet"""

    print("=" * 60)
    print("🎯 ENTRAÎNEMENT PROGRESSIF")
    print("=" * 60)

    best_map = -float('inf')  # Training loop
    config = run_config("config.yaml")

    for stage in list(config.keys())[1:]:  # First is backbone name
        # Configuration du stage
        config_stage = setup_training(model, config, stage)
        print(config_stage["title"])
        my_model.check_trainable_params(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=config_stage['lr'],
            momentum=0.9,
            weight_decay=0.0001
        )

        # Scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config_stage['step_size'],
            gamma=0.1,
            verbose=True
        )
        print(f"Optimizer: {len(trainable_params)} paramètres, LR={config_stage['lr']}")
        print(f"Training: {config_stage['epochs']} epochs")
        print("-" * 40)

        # Entraînement pour ce stage
        stage_losses = []
        for epoch in range(config_stage['epochs']):
            torch.cuda.empty_cache()
            epoch_loss = train_one_epoch(model, train_loader, device, optimizer, epoch=epoch)
            stage_losses.append(epoch_loss)

            lr_scheduler.step()

            results = evaluate(model, val_loader, device, epoch)
            current_map = results['map'].item()

            for k, v in results.items():
                print(f"\t{k}: {v.item()}")

            if current_map > best_map:
                best_map = current_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                }, os.path.join(save_dir, f'best_model_checkpoint_epoch_{stage}_{epoch}.pth'))
                best_stage = stage
                best_epoch_model = epoch
                print(f"\tModel saved at {best_stage} for epoch {best_epoch_model}")

        avg_stage_loss = sum(stage_losses) / len(stage_losses)
        print(f"\n🏁 {stage.upper()} terminé | Loss moyenne: {avg_stage_loss:.4f}\n")

    print("✨ ENTRAÎNEMENT PROGRESSIF TERMINÉ !")
    print(f"Meilleur stage at epoch: {best_stage, best_epoch_model}")

    return model, best_stage, best_epoch_model


def compute_iou(preds, targets, num_classes=2):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return sum(i for i in ious if not torch.isnan(torch.tensor(i))) / len(ious)


def alternative_training(model, train_loader, val_loader, device, save_dir):
    print("=" * 60)
    print("🎯 ENTRAÎNEMENT PROGRESSIF")
    print("=" * 60)

    config = run_config("config.yaml")
    best_val_iou = 0.0

    for stage in list(config.keys())[1:]:  # First is backbone name
        # Configuration du stage
        config_stage = setup_training(model, config, stage)
        print(config_stage["title"])
        my_model.check_trainable_params(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trainable_params, lr=config_stage['lr'])

        model.train()
        for epoch in range(config_stage['epochs']):
            header = f"Training Epoch {epoch}:"
            running_loss = 0.0
            with tqdm(train_loader, desc=header) as tq:
                for i, (images, masks) in enumerate(tq):
                    images = [image.to(device) for image in images]
                    masks = [mask.to(device) for mask in masks]

                    optimizer.zero_grad()
                    outputs = model(images)['out']

                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            val_iou_total = 0.0
            with torch.no_grad():
                with tqdm(val_loader, desc="Validation") as tq:
                    for i, (images, masks) in enumerate(tq):
                        images = images.to(device)
                        masks = masks.to(device)

                        outputs = model(images)['out']
                        val_iou_total += compute_iou(outputs, masks)
            avg_val_iou = val_iou_total / len(val_loader)

            print(f"Epoch {epoch}/{config_stage['epochs']} | Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            if avg_val_iou > best_val_iou:
                best_val_iou = avg_val_iou
                torch.save(model.state_dict(), f"{save_dir}/best_deeplabv3_rooftop.pth")
                print(f"✅ Nouveau meilleur modèle sauvegardé (IoU = {best_val_iou:.4f})")


if __name__ == "__main__":
    # our dataset has two classes only - background and rooftop
    num_classes = 2
    MODE_EXP = "masks"
    # use our dataset and defined transformations

    mean_pts_per_polygone = my_data_processing.get_mean_pts_per_poly(os.path.join(cfg.BASE_PATH, "train", "labels"))

    dataset_train = my_dataset.CustomDataset(subset="train", mode=MODE_EXP, mean_pts=mean_pts_per_polygone,
                                  transforms=get_transform(True))
    dataset_valid = my_dataset.CustomDataset(subset="valid", mode=MODE_EXP, mean_pts=mean_pts_per_polygone,
                                  transforms=get_transform(False))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=4,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=4,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=2,
        pin_memory=True
    )

    model = my_model.get_model(num_classes, mode=MODE_EXP).to(cfg.DEVICE)

    model, best_stage, best_epoch_model = progressive_training(model, train_data_loader, val_data_loader, cfg.DEVICE, save_dir="checkpoints")
    # alternative_training(model, train_data_loader, val_data_loader, cfg.DEVICE, save_dir="checkpoints")

    exit()

    checkpoint_path = os.path.join("checkpoints", f"best_model_checkpoint_epoch_stage1_{118}.pth")

    def load_model(checkpoint_path):
        model = my_model.get_model(num_classes, mode=MODE_EXP)
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(cfg.DEVICE)
        model.eval()
        return model


    model = load_model(checkpoint_path)

    dataset_test = my_dataset.CustomDataset(subset="test", mode=MODE_EXP, mean_pts=mean_pts_per_polygone,
                                  transforms=get_transform(train=False))
    batch_size_test = 2

    # define training and validation data loaders
    test_data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size_test,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    outputs = []
    real_images = []
    true_outputs = []
    with tqdm(test_data_loader, desc="Test dataloader inference") as tq:
        for i, (images, targets) in enumerate(tq):
            images = [img.to(cfg.DEVICE) for img in images]
            targets = [{k: v.to(cfg.DEVICE) for k, v in target.items()} for target in targets]

            with torch.no_grad():
                output = model(images)
            outputs.append(output)
            real_images.append(images)
            true_outputs.append(targets)

    rd_idx = np.random.randint(0, len(real_images))
    rd_idx_bis = np.random.randint(0, 1)
    image_1 = real_images[rd_idx][rd_idx_bis]
    sample_pred_target = outputs[rd_idx][rd_idx_bis]
    sample_true_target = true_outputs[rd_idx][rd_idx_bis]

    image_2 = image_1.clone()

    threshold = 0.7

    pred_masks = sample_pred_target["masks"].bool()
    pred_scores = sample_pred_target["scores"]
    for mask, score in zip(pred_masks, pred_scores):
        if score > threshold:
            image_1 = draw_segmentation_masks(image_1, mask, alpha=0.3, colors="red")

    true_masks = sample_true_target["masks"].bool()
    for mask in true_masks:
        image_2 = draw_segmentation_masks(image_2, mask, alpha=0.3, colors="red")

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].set_title("Pred")
    axes[0].imshow(image_1.cpu().permute(1, 2, 0))
    axes[0].axis("off")

    axes[1].set_title("True")
    axes[1].imshow(image_2.cpu().permute(1, 2, 0))
    axes[1].axis("off")
    plt.show()