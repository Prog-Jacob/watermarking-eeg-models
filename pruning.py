import torch
import torch.nn.utils.prune as prune


def compute_descending_mask(self, t, default_mask):
    tensor_size = t.nelement()
    nparams_toprune = round(self.amount * tensor_size)

    mask = default_mask.clone(memory_format=torch.contiguous_format)

    if nparams_toprune != 0:
        topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=True)
        mask.view(-1)[topk.indices] = 0

    return mask


class Pruning:
    def random():
        return prune.random_unstructured

    def ascending():
        return prune.l1_unstructured

    def descending():
        prune.L1Unstructured.compute_mask = compute_descending_mask
        return prune.l1_unstructured
