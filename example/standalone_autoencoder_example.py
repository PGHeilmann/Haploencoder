import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import Custom Classes
import classes as cs

# --------------------
# Config
# --------------------
DATA_DIR = "ae"  # directory containing the data files below
MARKERS_FILE = "hybrids.txt"         # space-separated marker data
BLOCKS_FILE = "LD07FL.csv"           # tab-separated, has column "Markers" that contain block assignment
TARGETS_FILE = "Y_data.csv"          # space-separated, column "GY" in the example
CV_INDEX_FILE = "TrainTestIndex.csv" # space-separated, each col is a CV iteration

BATCH_SIZE = 64
LR = 1e-3
L2_LAMBDA = 1e-3
NUM_EPOCHS = 10
NUM_ITERS = 100

OUT_DIR = "."  # outputs saved here


# --------------------
# Load data
# --------------------
markers_path = os.path.join(DATA_DIR, MARKERS_FILE)
blocks_path = os.path.join(DATA_DIR, BLOCKS_FILE)
targets_path = os.path.join(DATA_DIR, TARGETS_FILE)

marker_df = pd.read_csv(markers_path,  sep=" ")
blocks_df = pd.read_csv(blocks_path,   sep="\t")
targets_df = pd.read_csv(targets_path, sep=" ")

# Filter out blocks consisting of one marker
is_single = blocks_df["Markers"].str.count(";") == 1
blocks_df = blocks_df[~is_single].reset_index(drop=True)

# Required format: -1, 0, 1 (input is {0,1,2})
marker_df = marker_df - 1

# Build per-block tensors
block_tensors = []
input_sizes = []
for i in range(len(blocks_df)):
    iter_markers = blocks_df["Markers"].iloc[i].split(";")
    iter_markers = [m.replace("-", ".") for m in iter_markers if m]  # drop empty strings
    input_sizes.append(len(iter_markers))

    block_data = marker_df[iter_markers].to_numpy()
    block_tensors.append(torch.tensor(block_data, dtype=torch.float32))

marker_tensor = torch.tensor(marker_df.to_numpy(), dtype=torch.float32)
output_size = marker_df.shape[1]

# CV indices
idx_df = pd.read_csv(CV_INDEX_FILE, sep=" ")

# Targets (assume first column is GY)
y_all = targets_df.iloc[:, 3].to_numpy().astype(np.float32)


# --------------------
# Training loop (CPU)
# --------------------

# it = 0
for it in range(NUM_ITERS):
    print(f"This is iteration {it + 1}/{NUM_ITERS}")
    iter_mask = idx_df.iloc[:, it].to_numpy().astype(bool)  # True = Test set

    # Mask labels for test set entries (set to NaN)
    y_temp = y_all.copy()
    y_temp[iter_mask] = np.nan
    y_tensor = torch.tensor(y_temp, dtype=torch.float32)

    dataset = cs.SemiSupervisedDataset(block_tensors, marker_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = cs.SSAutoencoder(input_sizes, output_size=output_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    recon_loss = nn.functional.mse_loss
    sup_loss = nn.functional.mse_loss
    corr_loss = cs.CorrelationLoss()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        model.train()

        for batch_inputs, batch_outputX, batch_targets in dataloader:
            optimizer.zero_grad()

            labeled_mask = ~torch.isnan(batch_targets)  # True where label exists
            out_X, out_Y = model(batch_inputs)
            out_Y = out_Y.view(-1)  # no hard-coded length

            # Losses
            l_sup = sup_loss(out_Y[labeled_mask], batch_targets[labeled_mask])
            l_recon = recon_loss(out_X, batch_outputX)
            l_corr = corr_loss(out_Y[labeled_mask], batch_targets[labeled_mask])

            # L2 penalty on supervised head
            l2_penalty = torch.tensor(0.0)
            for param in model.supervised_head.parameters():
                l2_penalty = l2_penalty + torch.norm(param).pow(2)

            loss = l_corr + (l_sup + L2_LAMBDA * l2_penalty) + l_recon
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        epoch_loss = running_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}")

        with torch.no_grad():
            model.eval()
            _, preds = model(block_tensors)
            preds = preds.view(-1).numpy()
            corr = np.corrcoef(preds[iter_mask], y_all[iter_mask])[1, 0]
            print(f"Test correlation: {corr:.4f}")

    # --------------------
    # Exports for this iteration
    # --------------------
    with torch.no_grad():
        model.eval()
        _, preds = model(block_tensors)
        preds = preds.view(-1)

        # Encodings per block for all observations
        all_inputs = [blk.unsqueeze(0) for blk in block_tensors]  # add batch dim
        encoded_blocks = [enc(inp) for inp, enc in zip(all_inputs, model.encoders)]

        combined_encoded = torch.cat(encoded_blocks, dim=0)              # [num_blocks, N]
        combined_encoded = combined_encoded.view(combined_encoded.size(0), -1).t()  # [N, num_blocks]

        # Keep your original separators
        np.savetxt(os.path.join(OUT_DIR, f"Encoded_Blocks_{it + 1}.txt"),
                   combined_encoded.numpy(), delimiter=";")
        np.savetxt(os.path.join(OUT_DIR, f"Predictions_{it + 1}.txt"),
                   preds.numpy(), delimiter=";")

        # First row weights, following rows: node outputs per observation
        weights = model.supervised_head.weight.detach()  # [1, num_blocks]
        concatenated = torch.cat((weights, combined_encoded), dim=0)

        with open(os.path.join(OUT_DIR, f"WeightsAndBlockLayer_{it + 1}.txt"), "w") as file:
            for row in concatenated:
                row_str = " ".join(map(str, row.tolist()))
                file.write(row_str + "\n")

        torch.save(model, os.path.join(OUT_DIR, f"SS_Model_{it + 1}.pth"))


