"""Weight pruning strategies (random, ascending, descending magnitude)."""

import torch
import torch.nn.utils.prune as prune


class _DescendingL1Unstructured(prune.L1Unstructured):
    """Prune the highest-magnitude weights (the inverse of L1Unstructured)."""

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = round(self.amount * tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=True)
            mask.view(-1)[topk.indices] = 0

        return mask


def _descending_unstructured(module, name, amount, importance_scores=None):
    _DescendingL1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


class Pruning:
    """Factory of unstructured pruning functions, keyed by `--pruning_method`."""

    @staticmethod
    def random():
        """Prune random weights."""
        return prune.random_unstructured

    @staticmethod
    def ascending():
        """Prune the lowest-magnitude weights first."""
        return prune.l1_unstructured

    @staticmethod
    def descending():
        """Prune the highest-magnitude weights first."""
        return _descending_unstructured
