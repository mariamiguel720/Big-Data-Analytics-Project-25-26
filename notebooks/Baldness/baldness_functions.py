from scipy.ndimage import uniform_filter, sobel as scipy_sobel
import numpy as np
from PIL import Image
import io
import os

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql import Row

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pyarrow.parquet as pq
from pyspark.ml.torch.distributor import TorchDistributor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)



# FUNCTIONS SECTION 9.

def local_std(zone, win=16):
    """Computes local texture roughness (pixel-wise local standard deviation)."""
    mean_sq  = uniform_filter(zone ** 2, size=win)
    sq_mean  = uniform_filter(zone,      size=win) ** 2
    variance = np.maximum(mean_sq - sq_mean, 0)
    return float(np.sqrt(variance).mean())

def sobel_magnitude(zone):
    """Computes mean Sobel edge magnitude — detects edges regardless of skin colour."""
    sx = scipy_sobel(zone, axis=1)
    sy = scipy_sobel(zone, axis=0)
    return float(np.sqrt(sx**2 + sy**2).mean())

def extract_texture_features(rows, EPS = 1e-6):
    """
    Extracts texture and gradient features from image binary content.
    Processed partition by partition (mapPartitions) — imports done once per partition.

    Strategy: texture + gradient instead of absolute colour values.
    This makes features robust to skin tone, avoiding misclassification of
    dark-skinned bald individuals whose smooth scalp may resemble dark hair.

    Image zones (assumes images resized to IMG_SIZE x IMG_SIZE):
      - top    : upper 40% of image — should be smooth if bald
      - sides  : lower 30% left+right — lateral hair region
      - transition: horizontal band centred at top/sides boundary

    Extracted features:
      [0] std_local_topo       : mean local texture roughness at the top (8x8 window)
      [1] std_local_lados      : mean local texture roughness at the sides
      [2] ratio_texture        : std_local_topo / (std_local_lados + eps) — PRIMARY feature
      [3] sobel_topo           : mean edge intensity at the top (hair creates edges)
      [4] sobel_lados          : mean edge intensity at the sides
      [5] ratio_sobel          : sobel_topo / (sobel_lados + eps)
      [6] gradient_transition  : vertical gradient at the top/sides boundary
                                 (bald: abrupt transition; not bald: gradual)
      [7] std_topo_global      : global std of the top zone (complements local measure)
    """
    for row in rows:
        try:
            img_bytes = row["content"]
            if img_bytes is None or len(img_bytes) < 1000:
                continue

            # Convert image bytes to grayscale numpy array
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
            arr = np.array(img, dtype=np.float32)
            h, w = arr.shape

            # Zone definition
            top_h  = int(h * 0.40)
            side_h = int(h * 0.30)
            side_w = int(w * 0.30)

            zone_top   = arr[:top_h, :]
            zone_left  = arr[top_h:top_h + side_h, :side_w]
            zone_right = arr[top_h:top_h + side_h, w - side_w:]
            zone_sides = np.concatenate([zone_left, zone_right], axis=1)

            # Transition band: ±10 pixels around the top/sides boundary
            trans_band      = 10
            y0              = max(0, top_h - trans_band)
            y1              = min(h, top_h + trans_band)
            zone_transition = arr[y0:y1, :]

            # Texture features (local std) 
            std_local_topo  = local_std(zone_top)
            std_local_lados = local_std(zone_sides)
            ratio_texture   = std_local_topo / (std_local_lados + EPS)

            # Gradient features (Sobel)
            sobel_topo  = sobel_magnitude(zone_top)
            sobel_lados = sobel_magnitude(zone_sides)
            ratio_sobel = sobel_topo / (sobel_lados + EPS)

            # Transition gradient
            # Bald: abrupt vertical gradient at boundary (high value)
            # Not bald with short hair: gradual transition (low value)
            gy                  = np.abs(np.diff(zone_transition, axis=0)).mean()
            gradient_transition = float(gy)

            # Global top std (complementary to local measure)
            std_topo_global = float(zone_top.std())

            features = [
                std_local_topo,
                std_local_lados,
                ratio_texture,
                sobel_topo,
                sobel_lados,
                ratio_sobel,
                gradient_transition,
                std_topo_global,
            ]

            yield Row(path=row["path"], label=row["label"], features_raw=features)

        except Exception:
            continue  # skip unreadable or corrupt images silently



# Compute squared Euclidean distance from each image to its class centroid
# The centroid is broadcast to all workers to avoid repeated serialization — big-data-safe
# NOTE: UDFs are opaque to Spark's query optimizer. Here they are used because
# MLlib's Vectors.squared_distance is not natively available as a DataFrame
# function. The broadcast of the centroid ensures the UDF is efficient at scale.
def make_dist_udf(centroid, spark, Vectors):
    """ Returns a UDF that computes squared distance from the given centroid.
        The centroid is broadcast to all workers for efficiency.   
    Parameters:
        centroid: The centroid vector to compute distances from.
        spark: The SparkSession, used to create the broadcast variable.
        Vectors: The MLlib Vectors module, used for distance computation.
    Returns:
        A UDF that takes a feature vector and returns its squared distance to the centroid.
    """
    centroid_bc = spark.sparkContext.broadcast(centroid)

    @udf(DoubleType())
    def _udf(features):
        if features is None:
            return None
        return float(Vectors.squared_distance(features, centroid_bc.value))

    return _udf


# FUNCTIONS SECTION 10 - Save to split

def save_split_to_disk(df, split_name, OUTPUT_DIR):
    """
    Saves all images in a Spark DataFrame to disk under OUTPUT_DIR/{split_name}/{label}/.

    NOTE: collect() is used here to bring image content to the driver for local disk writing.
    This is NOT big-data-safe — in production, images would be written directly to
    distributed storage (S3, GCS, HDFS) using df.write, without collecting to the driver.
    It is used here because the studio environment only supports local file writes.

    Args:
        df         : Spark DataFrame with columns [path, content, label]
        split_name : one of 'train', 'val', 'test'

    Returns:
        tuple (saved, errors) — count of successfully saved and failed images
    """
    rows   = df.select("path", "content", "label").collect()
    saved  = 0
    errors = 0

    for row in rows:
        try:
            filename = os.path.basename(row["path"])
            out_path = f"{OUTPUT_DIR}/{split_name}/{row['label']}/{filename}"
            with open(out_path, "wb") as f:
                f.write(row["content"])
            saved += 1
        except Exception as e:
            errors += 1

    return saved, errors



# FUNCTIONS SECTION 11 

# COLOUR MODE INSPECTION
# Inspect the colour mode of all images before any conversion.
# This determines whether a conversion step is necessary.

def inspect_colour_mode(rows):
    """
    Processed per partition (mapPartitions) - imports done ONCE per partition.
    Reads each image and returns its colour mode and channel count.
    No conversion is performed at this stage.
    """
    for row in rows:
        try:
            img_bytes = row["content"]
            if img_bytes is None or len(img_bytes) < 1000:
                continue

            img = Image.open(io.BytesIO(img_bytes))

            mode_to_channels = {
                "L":    1,
                "RGB":  3,
                "RGBA": 4,
                "CMYK": 4,
                "P":    1,
                "LA":   2,
            }

            yield Row(
                path=row["path"],
                label=row["label"],
                original_split=row["original_split"],
                original_mode=img.mode,
                original_n_channels=mode_to_channels.get(img.mode, 3),
            )

        except Exception:
            continue


# Resize all images to 224x224 and apply ImageNet normalisation.
# df_processed is kept as a lazy DataFrame - no cache, no Parquet.
# The Spark engine will compute it on demand when augmentation runs.
# Storing float32 arrays (~600KB each) for 13k images would exceed
# the Spark JVM heap - lazy evaluation keeps memory usage bounded.
def resize_and_normalise(rows, IMAGENET_MEAN, IMAGENET_STD):
    """
    Processed per partition (mapPartitions) - imports done ONCE per partition.

    For each image:
      1. Resize to 224x224 using LANCZOS (best quality downsampling)
      2. Normalise pixel values to [0, 1]
      3. Apply ImageNet per-channel mean/std normalisation
      4. Serialise the resulting float32 array back to bytes
    """
    for row in rows:
        try:
            img_bytes = row["content"]
            if img_bytes is None or len(img_bytes) < 1000:
                continue

            img = Image.open(io.BytesIO(img_bytes))
            img_resized = img.resize((224, 224), Image.LANCZOS)

            # Normalise to [0,1] then apply ImageNet mean/std per channel
            # This matches the input distribution expected by the EfficientNet-B0 backbone
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

            yield Row(
                path=row["path"],
                label=row["label"],
                label_index=1 if row["label"] == "Bald" else 0,  # Bald=1, NotBald=0
                original_split=row["original_split"],
                image_array=arr.tobytes(),
                image_shape="224x224x3",
            )

        except Exception:
            continue

def augment_partition(rows, _IMAGENET_MEAN, _IMAGENET_STD, N_AUGMENTS):
    """
    Runs on WORKERS (never on the driver).
    Imports done ONCE per partition -- efficient at Big Data scale.

    Logic:
      - Every image is always emitted unchanged (original always yielded)
      - Augmentation applied ONLY to: label == "Bald" AND original_split == "train"
      - N_AUGMENTS augmented versions generated per eligible image

    Augmentation pipeline:
      - ImageNet normalisation is reversed before augmentation (PIL requires uint8)
      - Augmented image is re-normalised with ImageNet mean/std after transforms
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
    import random

    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(_IMAGENET_STD,  dtype=np.float32)

    def apply_augmentation(img):
        # 1. Horizontal flip -- baldness is symmetric (50% prob)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Rotation +-15 degrees -- simulates tilted camera (60% prob)
        if random.random() > 0.4:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 3. Brightness jitter +-20% -- simulates lighting variation (60% prob)
        if random.random() > 0.4:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        # 4. Contrast jitter +-20% (60% prob)
        if random.random() > 0.4:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        # 5. Zoom/crop -- simulates different distances to subject (50% prob)
        if random.random() > 0.5:
            w, h  = img.size
            zoom  = random.uniform(0.85, 1.0)
            left  = int(w * (1 - zoom) / 2)
            top   = int(h * (1 - zoom) / 2)
            img   = img.crop((left, top, w - left, h - top)).resize((w, h), Image.LANCZOS)

        # 6. Gaussian blur - robustness to slightly out-of-focus images (30% prob)
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        return img

    for row in rows:
        # Always emit the original -- every image passes through unchanged
        yield row

        # Augmentation only for Bald images in the train split
        if row["label"] != "Bald" or row["original_split"] != "train":
            continue

        try:
            arr_orig = np.frombuffer(row["image_array"], dtype=np.float32).reshape(224, 224, 3)

            # Reverse ImageNet normalisation before converting to PIL (requires uint8 in [0, 255])
            arr_uint8 = ((arr_orig * std + mean) * 255).clip(0, 255).astype(np.uint8)
            img_orig  = Image.fromarray(arr_uint8, mode="RGB")

            for i in range(N_AUGMENTS):
                arr_aug = np.array(apply_augmentation(img_orig.copy()), dtype=np.float32) / 255.0

                # Re-apply ImageNet normalisation after augmentation
                arr_aug = (arr_aug - mean) / std

                yield Row(
                    path=row["path"] + f"_aug{i}",
                    label=row["label"],
                    label_index=row["label_index"],
                    original_split=row["original_split"],
                    image_array=arr_aug.tobytes(),
                    image_shape="224x224x3",
                )

        except Exception:
            continue



# FUNCTIONS SECTION 12 

# Reads images from the augmented Parquet file and returns
# PyTorch tensors ready for training.
# PyArrow is used to read Parquet directly without Spark --
# this avoids the OOM issues caused by collecting large float32
# arrays through the Spark JVM heap.
class BaldDataset(Dataset):
    def __init__(self, parquet_path, split):
        """
        Reads from Parquet and filters by split ("train", "validation", "test").
        Arrays are already normalised with ImageNet mean/std, shape (224, 224, 3).
        """
        table = pq.read_table(
            parquet_path,
            filters=[("original_split", "=", split)],
            columns=["image_array", "label_index"]
        )
        self.images = table["image_array"].to_pylist()
        self.labels = table["label_index"].to_pylist()
        print(f"  {split}: {len(self.labels):,} images")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr    = np.frombuffer(self.images[idx], dtype=np.float32).reshape(224, 224, 3)
        # HWC -> CHW (PyTorch expects channels first)
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label
    

# MODEL DEFINITIONS
# Defines build functions for all three models.
# All functions follow the same interface:
#   frozen=True  -> backbone frozen (Phase 1)
#   frozen=False -> backbone unfrozen (Phase 2)

def build_efficientnet(frozen: bool = True) -> nn.Module:
    """
    Builds an EfficientNet-B0 with a binary classification head.

    Classification head: Dropout(0.3) -> Linear(1280, 1)
    Output: single logit per image.
    BCEWithLogitsLoss is applied outside the model during training.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    in_features      = model.classifier[1].in_features  # 1280 in EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


def build_resnet(frozen: bool = True) -> nn.Module:
    """
    Builds a ResNet-50 with a binary classification head.

    Classification head: Dropout(0.3) -> Linear(2048, 1)
    Output: single logit per image.
    BCEWithLogitsLoss is applied outside the model during training.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features  # 2048 in ResNet-50
    model.fc    = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


def build_mobilenet(frozen: bool = True) -> nn.Module:
    """
    Builds a MobileNetV3-Large with a binary classification head.

    Classification head: Dropout(0.3) -> Linear(960, 1)
    Output: single logit per image.
    BCEWithLogitsLoss is applied outside the model during training.
    """
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    in_features      = model.classifier[0].in_features  # 960 in MobileNetV3-Large
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


def count_params(m):
    """ Utility function to count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    return trainable, total


# TRAINING AND EVALUATION FUNCTIONS
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one full training epoch over the provided DataLoader.

    Sets the model to training mode, which enables Dropout and
    gradient computation. Gradients are reset at the start of
    each batch to avoid accumulation across batches.

    Args:
        model     : PyTorch model to train
        loader    : DataLoader for the training set
        criterion : loss function (BCEWithLogitsLoss)
        optimizer : Adam optimizer
        device    : torch.device (cuda or cpu)

    Returns:
        avg_loss (float) : mean loss across all batches in the epoch
    """
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.float().to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze(1)   # (B, 1) -> (B,)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the provided DataLoader.

    Sets the model to eval mode, which disables Dropout.
    Gradients are disabled entirely via torch.no_grad(),
    saving memory and computation.

    Metrics computed:
        loss      — BCEWithLogitsLoss averaged over all batches
        accuracy  — fraction of correctly classified images
        precision — of all predicted Bald, how many are truly Bald
        recall    — of all truly Bald, how many were correctly found
        f1        — harmonic mean of precision and recall
        auc       — AUC-ROC, separation quality independent of threshold

    Args:
        model     : PyTorch model to evaluate
        loader    : DataLoader for the validation or test set
        criterion : loss function (BCEWithLogitsLoss)
        device    : torch.device (cuda or cpu)

    Returns:
        dict with keys: loss, accuracy, precision, recall, f1, auc
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for X, y in loader:
            X, y   = X.to(device), y.float().to(device)
            logits = model(X).squeeze(1)
            probs  = torch.sigmoid(logits)       # logit -> probability [0, 1]
            preds  = (probs >= 0.5).long()       # probability -> binary prediction

            total_loss  += criterion(logits, y).item()
            all_probs   += probs.cpu().tolist()
            all_preds   += preds.cpu().tolist()
            all_labels  += y.long().cpu().tolist()

    avg_loss = total_loss / len(loader)

    return {
        "loss"      : avg_loss,
        "accuracy"  : accuracy_score(all_labels, all_preds),
        "precision" : precision_score(all_labels, all_preds, zero_division=0),
        "recall"    : recall_score(all_labels, all_preds,    zero_division=0),
        "f1"        : f1_score(all_labels, all_preds,        zero_division=0),
        "auc"       : roc_auc_score(all_labels, all_probs),
    }




# PHASE 1: FEATURE EXTRACTION (FROZEN BACKBONE)
def run_phase1(
    augmented_path:  str,
    checkpoint_path: str,
    model_name:      str,
    lr:              float = 1e-3,
    epochs:          int   = 10,
    batch_size:      int   = 32,
    num_workers:     int   = 4,
    early_stop_pat:  int   = 3,
):
    """
    Phase 1: train only the classification head with the backbone fully frozen.

    The backbone weights inherited from ImageNet are preserved entirely.
    Only the classification head is updated, allowing it to converge quickly
    before any backbone weights are modified in Phase 2.

    All imports are inside this function because TorchDistributor serialises
    it and sends it to a Spark executor, which does not have access to the
    notebook's namespace.

    Saves the best checkpoint (highest val_f1) to checkpoint_path.

    Args:
        augmented_path  : path to the augmented Parquet file
        checkpoint_path : path to save the best Phase 1 checkpoint
        model_name      : one of "efficientnet", "resnet50", "mobilenet"
        lr              : learning rate for the Adam optimiser
        epochs          : maximum number of training epochs
        batch_size      : number of images per batch
        num_workers     : number of DataLoader worker processes
        early_stop_pat  : epochs without improvement before stopping

    Returns:
        dict with the best epoch's metrics: loss, accuracy, precision,
        recall, f1, auc.
    """

    # Library imports inside the function for Spark executor compatibility
    # These imports are executed once per Spark executor when the function is deserialised,
    # not at the time of definition in the notebook. 
    # These imports are mandatory for the function to run on the Spark workers, which do not have access to the notebook's global namespace.
    import os
    import torch
    import torch.nn as nn
    from torchvision import models
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    model_builders = {
        "efficientnet": build_efficientnet,
        "resnet50":     build_resnet,
        "mobilenet":    build_mobilenet,
    }
    if model_name not in model_builders:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Choose from: {list(model_builders.keys())}")

    model = model_builders[model_name](frozen=True).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 [{model_name}] — trainable: {trainable:,} / {total:,} parameters")

    # Data
    train_ds = BaldDataset(augmented_path, "train")
    val_ds   = BaldDataset(augmented_path, "validation")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5,
    )

    # Training loop
    best_f1       = 0.0
    best_metrics  = {}
    epochs_no_imp = 0

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1 [{model_name}] — Frozen backbone ({epochs} epochs, LR={lr})")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):

        train_loss  = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"accuracy: {val_metrics['accuracy']:.4f} | "
            f"precision: {val_metrics['precision']:.4f} | "
            f"recall: {val_metrics['recall']:.4f} | "
            f"f1: {val_metrics['f1']:.4f} | "
            f"auc: {val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1       = val_metrics["f1"]
            best_metrics  = val_metrics
            epochs_no_imp = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Checkpoint saved (val_f1={best_f1:.4f})")
        else:
            epochs_no_imp += 1
            print(f"  -> No improvement ({epochs_no_imp}/{early_stop_pat})")
            if epochs_no_imp >= early_stop_pat:
                print(f"  -> Early stopping triggered at epoch {epoch}")
                break

    print(f"\nPhase 1 [{model_name}] complete. Best val_f1: {best_f1:.4f}")
    return best_metrics



# PHASE 2: FINE-TUNING
def run_phase2(
    augmented_path:      str,
    phase1_ckpt_path:    str,
    phase2_ckpt_path:    str,
    model_name:          str,
    blocks_to_unfreeze:  list,
    lr:                  float = 1e-4,
    epochs:              int   = 15,
    batch_size:          int   = 32,
    num_workers:         int   = 4,
    early_stop_pat:      int   = 4,
):
    """
    Phase 2: load the best Phase 1 checkpoint, unfreeze selected backbone
    blocks, and fine-tune with a lower learning rate.

    All imports are inside this function because TorchDistributor serialises
    it and sends it to a Spark executor, which does not have access to the
    notebook's namespace.

    Saves the best checkpoint (highest val_f1) to phase2_ckpt_path.

    Args:
        augmented_path     : path to the augmented Parquet file
        phase1_ckpt_path   : path to the best Phase 1 checkpoint
        phase2_ckpt_path   : path to save the best Phase 2 checkpoint
        model_name         : one of "efficientnet", "resnet50", "mobilenet"
        blocks_to_unfreeze : list of block name substrings to unfreeze,
                             or None to unfreeze the full backbone
        lr                 : learning rate for the Adam optimiser
        epochs             : maximum number of training epochs
        batch_size         : number of images per batch
        num_workers        : number of DataLoader worker processes
        early_stop_pat     : epochs without improvement before stopping

    Returns:
        dict with the best epoch's metrics: loss, accuracy, precision,
        recall, f1, auc.
    """
    # Library imports inside the function for Spark executor compatibility
    # These imports are mandatory for the function to run on the Spark workers, which do not have access to the notebook's global namespace.
    import os
    import torch
    import torch.nn as nn
    from torchvision import models
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    model_builders = {
        "efficientnet": build_efficientnet,
        "resnet50":     build_resnet,
        "mobilenet":    build_mobilenet,
    }
    if model_name not in model_builders:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Choose from: {list(model_builders.keys())}")

    # Load Phase 1 checkpoint 
    model = model_builders[model_name](frozen=True).to(DEVICE)
    model.load_state_dict(torch.load(phase1_ckpt_path, map_location=DEVICE))
    print(f"Phase 1 checkpoint loaded for {model_name}.")

    # Unfreeze selected blocks
    if blocks_to_unfreeze is None:
        for param in model.parameters():
            param.requires_grad = True
        print("Phase 2 — full backbone unfrozen")
    else:
        for name, param in model.named_parameters():
            if any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True
        print(f"Phase 2 — unfrozen blocks: {blocks_to_unfreeze}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters")

    # Data
    train_ds = BaldDataset(augmented_path, "train")
    val_ds   = BaldDataset(augmented_path, "validation")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5,
    )

    # Training loop with early stopping
    best_f1       = 0.0
    best_metrics  = {}
    epochs_no_imp = 0

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2 [{model_name}] — Fine-tuning (max {epochs} epochs, LR={lr})")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):

        train_loss  = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"accuracy: {val_metrics['accuracy']:.4f} | "
            f"precision: {val_metrics['precision']:.4f} | "
            f"recall: {val_metrics['recall']:.4f} | "
            f"f1: {val_metrics['f1']:.4f} | "
            f"auc: {val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1       = val_metrics["f1"]
            best_metrics  = val_metrics
            epochs_no_imp = 0
            torch.save(model.state_dict(), phase2_ckpt_path)
            print(f"  -> Checkpoint saved (val_f1={best_f1:.4f})")
        else:
            epochs_no_imp += 1
            print(f"  -> No improvement ({epochs_no_imp}/{early_stop_pat})")
            if epochs_no_imp >= early_stop_pat:
                print(f"  -> Early stopping triggered at epoch {epoch}")
                break

    print(f"\nPhase 2 [{model_name}] complete. Best val_f1: {best_f1:.4f}")
    return best_metrics
