import torch
import torch.nn as nn


class TipletMiningLoss(nn.Module):
    '''
    Online Triplet Loss with hard mining. Takes the same input as if one was using the normal
    CrossEntropyLoss with PyTorch. The inputs are a batch of embeddings (N, embed_dim) and a vector (N)
    of corresponding labels. The forward function finds for each embedding the furthest positive and
    closest negative and calculates the triplet loss using those.

    Methods
    -------
        forward: vector_batch (N, embed_dim), labels_batch (N)
            Calculates the triplet loss and finds the furthest positives and closest negatives.

        get_positive_mask: labels (N)
            Finds the mask that contains valid positive matches. In other words it creates a binary
            mask of all inputs that have a matching label but ignores indices along the diagonal
            because that would be the same index as the current query vector.

        get_negative_mask: labels (N)
            Finds a mask that contains valid negative matches. In other words it creates a binary
            mask where two labels do not match.

    '''
    def __init__(self):
        super(TipletMiningLoss, self).__init__()

    def forward(self, vector_batch: torch.Tensor, labels_batch: torch.Tensor) -> torch.Tensor:

        # calculate all pairwise distances
        dists = torch.linalg.norm(vector_batch[:, None, :] - vector_batch[:, :] + 1e-8, dim=-1)

        # get the masks for valid positive matches and valid negative matches
        pos_mask = self.get_positive_mask(labels_batch)
        neg_mask = self.get_negative_mask(labels_batch)

        positive_dists = torch.max(dists * pos_mask, 1)[0]

        # min masking doesn't work because we'll just take 0 when the neg_mask is applied
        # so we take the maximum distance value and set that in the inverse of the negative mask
        # then when we take the min we won't accidentialy take a value where two indices match
        global_max_value = torch.max(dists).item()
        negative_dists = torch.min(dists + (global_max_value * ~neg_mask), 1)[0]

        # calculate triplet loss using mined pairs
        tl = torch.max(positive_dists - negative_dists + 0.5, torch.Tensor([0.0]))

        return torch.mean(tl)

    def get_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        # ones everywhere except for diagonal (same index)
        diag_mask = torch.eye(labels.size(0)).bool()
        diag_mask = ~diag_mask

        # same label
        equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

        # get the union of matching index and the diagonal mask
        mask = diag_mask & equal_mask

        return mask

    def get_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        # get the makes for where labels don't match
        return torch.ne(labels.unsqueeze(0), labels.unsqueeze(1))