from torch.utils.data import Dataset
import torch
import torch.nn as nn

### Custom Dataset

class SemiSupervisedDataset(Dataset):
    """
    Returns a tuple:
      ([block_i_row_tensor, ...], full_marker_row_tensor, target_value)
    blocksL: list of tensors [N, block_features]
    blocksV: tensor of all markers [N, total_markers]
    target:  tensor [N] with NaNs where label is missing, e.g. incomplete yield data
    """

    def __init__(self, blocksL, blocksV, target):
        super().__init__()
        self.blocksL = [block for block in blocksL]
        self.blocksV = blocksV
        self.target = target

    def __len__(self):
        return self.blocksV.shape[0]

    def __getitem__(self, idx):
        inputs = [block[idx] for block in self.blocksL]  # one row per block
        ae_output = self.blocksV[idx]
        target = self.target[idx]
        return inputs, ae_output, target


### Semisupervised NN architecture

class SSAutoencoder(nn.Module):
    """
    Requires a list where every element is the marker data belonging to one haploblock
    Returns (unsupervised output, supervised output) -> e.g. (reconstructed markers, yield)
    """
    def __init__(self, input_sizes, output_size=0, combined_hidden_size=1000):
        super().__init__()

        # Per-block encoders
        self.encoders = nn.ModuleList()
        for input_size in input_sizes:
            if input_size > 4:
                hidden = max(1, input_size // 2)
                self.encoders.append(
                    nn.Sequential(
                        nn.Linear(input_size, hidden, bias=False),
                        nn.LeakyReLU(negative_slope=0.1),
                        nn.Linear(hidden, 1, bias=False),
                    )
                )
            else:
                self.encoders.append(nn.Linear(input_size, 1, bias=False))

        # Unsupervised reconstruction path
        self.combined_fc = nn.Linear(len(input_sizes), combined_hidden_size)
        self.decoder = nn.Linear(combined_hidden_size, output_size)

        # Supervised prediction head
        self.supervised_head = nn.Linear(len(input_sizes), 1, bias=False)

        self._act = nn.LeakyReLU(negative_slope=0.1)
        self._tanh = nn.Tanh()

    def forward(self, x):
        # x: list of [B, features] tensors (one per block)
        encoded_blocks = [encoder(block) for block, encoder in zip(x, self.encoders)]
        combined = torch.cat(encoded_blocks, dim=1)  # [B, num_blocks]

        z = self._act(self.combined_fc(combined))
        unsup_out = self._tanh(self.decoder(z))      # [B, output_size]
        sup_out = self.supervised_head(combined)     # [B, 1]
        return unsup_out, sup_out


# Loss related functions
class CorrelationLoss(nn.Module):
    """
    Returns negative Pearson correlation: -corr(x, y).
    By minimizing the negative correlation, we maximize the positive correlation
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        corr_matrix = torch.corrcoef(torch.stack((x, y)))
        corr = corr_matrix[0, 1]
        return -corr
