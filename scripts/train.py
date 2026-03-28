import argparse
import logging
import random
import numpy as np
import torch
import collections
import torch.nn.functional as F
import wandb
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import FPLIIF
from src.utils.dataset import FaceSegDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "CelebAMaskHQ": {
        "num_classes": 19,
        "class_names": [
            "background", "skin", "nose", "eye_l", "eye_r", "eye_g",
            "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip",
            "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth",
        ]
    },
    "LaPa": {
        "num_classes": 11,
        "class_names": [
            "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
            "eye_g", "nose", "u_lip", "l_lip", "hair",
        ]
    },
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def edge_aware_ce_loss(
    logits: torch.Tensor, targets: torch.Tensor, device, lambda_edge: float = 1.0
):
    SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    SOBEL_Y = SOBEL_X.T

    # CE Loss
    ce_loss = F.cross_entropy(logits, targets, reduction="mean")

    with torch.no_grad():
        targets_float = targets.float().unsqueeze(1)

        sobel_x = SOBEL_X.view(1, 1, 3, 3).float().to(device)
        sobel_y = SOBEL_Y.view(1, 1, 3, 3).float().to(device)

        grad_x = F.conv2d(targets_float, sobel_x, padding=1)
        grad_y = F.conv2d(targets_float, sobel_y, padding=1)

        edge_strength = (grad_x**2 + grad_y**2).sqrt().squeeze(1)
        edge_mask = (edge_strength > 1e-3).float()

    # Edge CE Loss
    ce_edge = F.cross_entropy(logits, targets, reduction="none")
    edge_loss = (ce_edge * edge_mask).sum() / (edge_mask.sum() + 1e-7)

    loss = ce_loss + lambda_edge * edge_loss
    return ce_loss, loss

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss_accum = 0.0
    ce_loss_accum = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [train]", leave=False)
    for step, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)

        ce_loss, total_loss = edge_aware_ce_loss(
            logits, masks, device, lambda_edge=1.0
        )
        total_loss.backward()
        optimizer.step()

        total_loss_accum += total_loss.item()
        ce_loss_accum += ce_loss.item()
        
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    return total_loss_accum / len(loader), ce_loss_accum / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    total_loss_accum = 0.0
    ce_loss_accum = 0.0
    
    num_classes = config["num_classes"]
    class_f1_scores = {cls_idx: [] for cls_idx in range(num_classes)}

    pbar = tqdm(loader, desc="          [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        ce_loss, total_loss_val = edge_aware_ce_loss(
            logits, masks, device, lambda_edge=1.0
        )
        
        total_loss_accum += total_loss_val.item()
        ce_loss_accum += ce_loss.item()

        preds = logits.argmax(dim=1)

        for i in range(images.shape[0]):
            mask_gt   = masks[i].cpu().numpy()
            mask_pred = preds[i].cpu().numpy()
            
            # Evaluate every class f1
            for class_id in range(num_classes):
                if np.sum(mask_gt == class_id) > 0 or np.sum(mask_pred == class_id) > 0:
                    tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
                    fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
                    fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))
                    
                    precision = tp / (tp + fp + 1e-7)
                    recall    = tp / (tp + fn + 1e-7)
                    f1 = (2 * precision * recall) / (precision + recall + 1e-7)
                    
                    class_f1_scores[class_id].append(f1)

        pbar.set_postfix(loss=f"{total_loss_val.item():.4f}")

    # Calculate per-class averages
    f1_per_class = {}
    for cls_idx in range(num_classes):
        if len(class_f1_scores[cls_idx]) > 0:
            f1_per_class[cls_idx] = float(np.mean(class_f1_scores[cls_idx]))
        else:
            f1_per_class[cls_idx] = 0.0

    # Average mean f1 except background class
    foreground_f1s = [f1_per_class[cls_idx] for cls_idx in range(1, num_classes)]
    mean_f1 = float(np.mean(foreground_f1s))

    return (
        total_loss_accum / len(loader), 
        ce_loss_accum / len(loader), 
        mean_f1, 
        f1_per_class
    )

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    dataset_name = args.dataset_name
    assert dataset_name in DATASET_CONFIGS, dataset_name
    config = DATASET_CONFIGS[dataset_name]
    NUM_CLASSES = config["num_classes"]
    CLASS_NAMES = config["class_names"]

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.config.update({"dataset": dataset_name})

    train_ds = FaceSegDataset(str(Path(args.data_dir) / args.dataset_name), split="train")
    val_ds   = FaceSegDataset(str(Path(args.data_dir) / args.dataset_name), split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = FPLIIF(
        channel=args.channel, class_num=NUM_CLASSES,
        out_size=args.val_img_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")
    wandb.config.update({"total_params": total_params})

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.75, patience=3,
    )

    best_f1 = 0.0
    patience_ctr = 0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.decoder.out_size = 256
        train_loss, train_ce = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        model.decoder.out_size = args.val_img_size
        val_loss, val_ce, mean_f1, f1_per_class = evaluate(model, val_loader, device, config)
        scheduler.step(mean_f1)

        wandb_logs = {
            "epoch": epoch,
            "train/total_loss": train_loss,
            "train/ce_loss": train_ce,
            "val/total_loss": val_loss,
            "val/ce_loss": val_ce,
            "val/mF1": mean_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for cls_idx in range(NUM_CLASSES):
            wandb_logs[f"f1/{CLASS_NAMES[cls_idx]}"] = f1_per_class[cls_idx]
        wandb.log(wandb_logs)

        # Early stopping
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            patience_ctr = 0
            ckpt_path = save_dir / "best.pth"
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "mf1": best_f1},
                ckpt_path,
            )
            wandb.config.update({"best_epoch": epoch, "best_val_mf1": best_f1})
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}, no improvement for {args.patience} epochs"
                )
                break

    logger.info(f"Training done. Best mF1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for FPLIIF")
    parser.add_argument("--data_dir",      default="data")
    parser.add_argument("--dataset_name",  default="CelebAMaskHQ")
    parser.add_argument("--save_dir",      default="checkpoints")
    parser.add_argument("--wandb_project", default="AI6103")
    parser.add_argument("--run_name",      default=None)
    parser.add_argument("--channel",       type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=200)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=8e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--val_img_size",  type=int,   default=512)

    main(parser.parse_args())