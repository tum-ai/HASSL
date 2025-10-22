import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from distinctipy import get_colors, get_colormap
from torchvision.transforms import functional as TF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def make_robust_collate_fn(target_size: int = 224):
    """
    Collate that:
      - converts all tensors to float32 and contiguous
      - ensures 3 channels (repeat single-channel, trim >3 to first 3)
      - resizes each image to (target_size, target_size) with bilinear interpolation
      - returns batched tensor and metadata lists
    """
    def collate_fn(batch):
        imgs = []
        labels = []

        for b in batch:
            image, label = b
            t = image # expected tensor CHW
            # make contiguous and float32
            if not t.is_contiguous():
                t = t.contiguous()
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            # channel-handling
            if t.ndim != 3:
                raise RuntimeError(f"unexpected image ndim: {t.ndim}, expected 3 (C,H,W)")
            c, h, w = t.shape
            if c == 1:
                # repeat grayscale -> RGB
                t = t.repeat(3, 1, 1)
            elif c >= 3:
                # take first 3 channels
                if c > 3:
                    t = t[:3, :, :]
            # now t is 3 x H x W
            imgs.append(t)
            labels.append(label)

        # resize each to (3, target_size, target_size)
        resized = []
        for t in imgs:
            if t.shape[1] == target_size and t.shape[2] == target_size:
                resized.append(t)
            else:
                t4 = t.unsqueeze(0)  # 1,3,H,W
                t_res = F.interpolate(t4, size=(target_size, target_size), mode="bilinear", align_corners=False)
                resized.append(t_res.squeeze(0))
        batch_images = torch.stack(resized, dim=0)  # B,3,H,W

        return {
            "image": batch_images,
            "label": labels,
        }

    return collate_fn

from torchvision.transforms import functional as TF
from torch.utils.data import Subset

def _unwrap_dataset(ds):
    # peel off nested Subset wrappers to reach the real dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

class ImageOnlyToTensor:
    """Accepts (img, seg) or img; returns a CHW float tensor."""
    def __call__(self, sample):
        img = sample[0] if isinstance(sample, (tuple, list)) and len(sample) > 0 else sample
        if isinstance(img, torch.Tensor):
            return img
        return TF.to_tensor(img)

def extract_embeddings(
    model,
    dataset,
    device="cuda",
    batch_size=64,
    num_workers=4,
    target_size=224,
):
    collate_fn = make_robust_collate_fn(target_size=target_size)

    # --- override the BASE dataset's transform (not the Subset wrapper) ---
    base_ds = _unwrap_dataset(dataset)
    old_transform = getattr(base_ds, "transform", None)
    base_ds.transform = ImageOnlyToTensor()   # drop seg + to_tensor only

    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = model.eval().to(device)
        all_emb, all_labels = [], []

        print("Starting embedding extraction. Batches:", len(loader))
        with torch.inference_mode():
            for batch in tqdm(loader, desc="batches"):
                imgs = batch["image"].to(device, non_blocking=True)
                out = model(imgs)

                # normalize outputs
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    for k in ("output", "features", "global", "last_hidden_state"):
                        if k in out:
                            out = out[k]
                            break
                    else:
                        out = next(iter(out.values()))

                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(f"Model returned {type(out)} — expected Tensor")

                all_emb.append(out.detach().cpu().float().numpy())
                for lab in batch["label"]:
                    all_labels.append(str(lab))

        embeddings = np.vstack(all_emb)
        return embeddings, all_labels

    finally:
        # always restore the original transform
        base_ds.transform = old_transform


def compute_tsne_and_plot(embeddings, labels, out_path="tsne.png",
                          pca_dim=50, tsne_perplexity=30, random_state=0,
                          max_legend=25, base_cmap_name="hsv"):
    """
    Compute PCA (optional) -> t-SNE and save a scatter plot with one unique color per label.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_samples, n_features)
    labels     : iterable of label ids/names (length n_samples)
    out_path   : str, where to save the PNG
    pca_dim    : int or None, reduce to this dim before t-SNE if < original dim
    tsne_perplexity : int, t-SNE perplexity (auto-clamped to valid range)
    random_state : int, RNG seed
    max_legend : int, max number of labels to show in legend (for readability)
    base_cmap_name : str, base cmap to sample for discrete colors (e.g., 'hsv', 'gist_ncar')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings)
    n, d = embeddings.shape

    # Optional PCA
    if pca_dim is not None and pca_dim < d:
        pca = PCA(n_components=min(pca_dim, d), random_state=random_state)
        emb_r = pca.fit_transform(embeddings)
    else:
        emb_r = embeddings

    # Perplexity must be < n; keep user's heuristic but clamp safely
    if n <= 2:
        raise ValueError("Need at least 3 points for t-SNE.")
    heuristic = max(5, n // 4 - 1)
    perplexity = min(tsne_perplexity, heuristic, n - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        n_jobs=8  # keep user's setting if their sklearn supports it
    )
    emb2 = tsne.fit_transform(emb_r)

    # Map labels -> integer indices
    unique_labels = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(unique_labels)}
    ints = np.array([label2idx[l] for l in labels], dtype=int)
    N = len(unique_labels)

    # Build a discrete colormap with exactly N colors (no interpolation)
    base = plt.cm.get_cmap(base_cmap_name, N)  # sample N evenly spaced colors
    cmap = ListedColormap(base(np.arange(N)))
    norm = BoundaryNorm(np.arange(-0.5, N + 0.5, 1), N)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(emb2[:, 0], emb2[:, 1], c=ints, cmap=cmap, norm=norm, s=6, alpha=0.85)

    # Legend: cap to max_legend entries for readability
    show_labels = unique_labels[:max_legend]
    handles = []
    for lab in show_labels:
        idx = label2idx[lab]
        color = cmap(idx)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=6, label=str(lab)))
    if len(show_labels) > 0:
        title = f"Labels (showing {len(show_labels)}/{N})" if N > max_legend else "Labels"
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize='small', title=title, borderaxespad=0.0)

    plt.title("t-SNE of embeddings")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Saved TSNE to", out_path)

    return emb2
