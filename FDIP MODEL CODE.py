# image_classify_final.py
# Usage examples:
#   python image_classify_final.py --data_dir "D:/FDIP MODEL TRAINING/Human activity" --epochs 10
#   python image_classify_final.py --data_dir "D:/FDIP MODEL TRAINING/Intruder detection system" --backbone efficientnet_b0 --batch_size 32 --unfreeze

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ---------------------------
# Helpers
# ---------------------------
def make_loaders(data_dir, img_size=224, batch_size=32, val_split=0.2, test_dir=None, num_workers=2, seed=42):
    """
    Supports two layouts:
     A) data_dir/{train,val,test}/{class}/img.jpg  -> uses as-is
     B) data_dir/train/{class}/img.jpg            -> auto-splits train into train/val

    If test_dir is provided, overrides data_dir/test.
    """
    # Transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    has_train = os.path.isdir(os.path.join(data_dir, "train"))
    has_val   = os.path.isdir(os.path.join(data_dir, "val"))
    has_test  = os.path.isdir(os.path.join(data_dir, "test")) or (test_dir and os.path.isdir(test_dir))

    if not has_train:
        # If the dataset path itself is a class-structured folder, treat it as train and split
        train_root = data_dir
    else:
        train_root = os.path.join(data_dir, "train")

    # Full "train" dataset
    full_train = datasets.ImageFolder(train_root, transform=train_tfms)
    class_names = full_train.classes

    # Build val/test
    if has_val:
        val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tfms)
    else:
        # Split from training data
        torch.manual_seed(seed)
        n_total = len(full_train)
        n_val = max(1, int(val_split * n_total))
        n_train = n_total - n_val
        train_set, val_set = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        # Ensure val uses eval transforms (not train augs)
        val_set = Subset(datasets.ImageFolder(train_root, transform=eval_tfms), val_set.indices)
        # Rebuild train_set to ensure it keeps train transforms
        train_set = Subset(full_train, train_set.indices)
        full_train = None  # free

    if has_test or test_dir:
        test_root = test_dir if test_dir else os.path.join(data_dir, "test")
        test_set = datasets.ImageFolder(test_root, transform=eval_tfms)
    else:
        test_set = None

    # If we had a split case, train_set already set; else use explicit train folder
    if not has_val:
        pass  # already created above
    else:
        train_set = datasets.ImageFolder(train_root, transform=train_tfms)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_set else None

    return train_loader, val_loader, test_loader, class_names


def build_model(backbone, num_classes, pretrained=True, unfreeze=False):
    """
    Supported backbones: resnet50, mobilenet_v2, efficientnet_b0
    """
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        feature_layers = [name for name, _ in model.named_parameters() if not name.startswith("fc.")]
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
        feature_layers = [name for name, _ in model.named_parameters() if not name.startswith("classifier.")]
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
        feature_layers = [name for name, _ in model.named_parameters() if not name.startswith("classifier.")]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Freeze or unfreeze
    for name, param in model.named_parameters():
        if unfreeze:
            param.requires_grad = True
        else:
            # freeze all except final layer
            if name in feature_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred
    }
    return metrics


def save_confusion_matrix(cm, class_names, out_path="confusion_matrix.png"):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Image Classification (Transfer Learning) — Final Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder with train[/val][/test] subfolders or class folders.")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional separate test folder path.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2, help="Used only when no explicit val folder exists.")
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze backbone for fine-tuning.")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = make_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_dir=args.test_dir,
        num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Model
    model = build_model(args.backbone, num_classes, pretrained=True, unfreeze=args.unfreeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    # Training loop with early stopping on val f1_macro
    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, f"best_{args.backbone}.pt")
    patience = args.early_stop_patience
    ticks = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        f1_macro = val_metrics["f1_macro"]
        msg = (f"[Epoch {epoch}/{args.epochs}] "
               f"TrainLoss={train_loss:.4f} | "
               f"Val Acc={val_metrics['accuracy']:.4f} F1_macro={f1_macro:.4f} "
               f"(Prec_macro={val_metrics['precision_macro']:.4f} Rec_macro={val_metrics['recall_macro']:.4f}) "
               f"[{(time.time()-t0):.1f}s]")
        print(msg)

        # Early stopping & checkpoint
        if f1_macro > best_f1:
            best_f1 = f1_macro
            ticks = 0
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names,
                        "backbone": args.backbone,
                        "img_size": args.img_size}, best_path)
            print(f"  ↳ Saved new best to: {best_path}")
        else:
            ticks += 1
            if ticks >= patience:
                print(f"Early stopping after no improvement for {patience} epoch(s).")
                break

    # Load best for evaluation
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["backbone"], len(ckpt["class_names"]), pretrained=False, unfreeze=True).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Final Validation report
    val_metrics = evaluate(model, val_loader, device)
    print("\n=== VALIDATION METRICS (Best) ===")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Precision (macro/weighted): {val_metrics['precision_macro']:.4f} / {val_metrics['precision_weighted']:.4f}")
    print(f"Recall    (macro/weighted): {val_metrics['recall_macro']:.4f} / {val_metrics['recall_weighted']:.4f}")
    print(f"F1        (macro/weighted): {val_metrics['f1_macro']:.4f} / {val_metrics['f1_weighted']:.4f}")
    print("\nClassification report (VAL):")
    print(classification_report(val_metrics["y_true"], val_metrics["y_pred"], target_names=class_names, zero_division=0))

    cm_val = confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"])
    save_confusion_matrix(cm_val, class_names, out_path=os.path.join(args.out_dir, "confusion_matrix_val.png"))
    print(f"Saved validation confusion matrix to {os.path.join(args.out_dir, 'confusion_matrix_val.png')}")

    # Optional Test report
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device)
        print("\n=== TEST METRICS (Best) ===")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision (macro/weighted): {test_metrics['precision_macro']:.4f} / {test_metrics['precision_weighted']:.4f}")
        print(f"Recall    (macro/weighted): {test_metrics['recall_macro']:.4f} / {test_metrics['recall_weighted']:.4f}")
        print(f"F1        (macro/weighted): {test_metrics['f1_macro']:.4f} / {test_metrics['f1_weighted']:.4f}")
        print("\nClassification report (TEST):")
        print(classification_report(test_metrics["y_true"], test_metrics["y_pred"], target_names=class_names, zero_division=0))

        cm_test = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
        save_confusion_matrix(cm_test, class_names, out_path=os.path.join(args.out_dir, "confusion_matrix_test.png"))
        print(f"Saved test confusion matrix to {os.path.join(args.out_dir, 'confusion_matrix_test.png')}")

    # Export label mapping for deployment
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx}\t{name}\n")
    print(f"Saved label map to {os.path.join(args.out_dir, 'labels.txt')}")


if __name__ == "__main__":
    main()
