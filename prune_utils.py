import torch
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import _validate_pruning_amount_init, _compute_nparams_toprune, _validate_pruning_amount
import time


class L2Unstructured(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t ** 2).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask
    
    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(L2Unstructured, cls).apply(
            module, name, amount=amount, importance_scores=importance_scores
        )

    
class L1Structured(prune.BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor based on their
    L\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, dim=-1):
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = 1
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a mask to apply on
        top of the ``default_mask`` by zeroing out the channels along the
        specified dim with the lowest L\ ``n``-norm.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied.  Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        norm = _compute_norm(t, self.n, self.dim)
        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(norm, k=nparams_tokeep, largest=True)
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask


    @classmethod
    def apply(cls, module, name, amount, dim, importance_scores=None):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super(L1Structured, cls).apply(
            module,
            name,
            amount=amount,
            dim=dim,
            importance_scores=importance_scores,
        )    
    
    
    


def get_child(name, model):
    children = list(model.named_children())
    res = [(name, model)]
    if children == []:
        return res
    else:
        for m in children:
            res += get_child(m[0], m[1])
        return res
    
def prune_amount(cur, target):
    # cur + (1 - cur) * amount = target
    amount = (target - cur) / (1 - cur)
    return amount


def prune_model_structured(params, cur_sparsity, target_sparsity):
    for n, m in params:
        prune.ln_structured(m, 'weight', amount=prune_amount(cur_sparsity, target_sparsity), n=1, dim=0)
    zero_weights = 0.0
    total_weights = 1e-8
    fc_zero_weights = 0.0
    fc_total_weights = 1e-8
    for n, m in params:
        zero_weights += torch.sum(m.weight == 0)
        total_weights += m.weight.nelement()
        if "fc" in n or "6" in n:
            fc_zero_weights += torch.sum(m.weight == 0)
            fc_total_weights += m.weight.nelement()

    conv_sparsity = (zero_weights - fc_zero_weights) / (total_weights - fc_total_weights)
    fc_sparsity = fc_zero_weights / fc_total_weights
    global_sparsity = zero_weights / total_weights
    return conv_sparsity, fc_sparsity, global_sparsity

def prune_model(params, cur_sparsity, target_sparsity, prune_function=prune.L1Unstructured):
    params_to_prune = [[f[1], 'weight'] for f in params]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune_function,
        amount=prune_amount(cur_sparsity, target_sparsity),
    )
    zero_weights = 0.0
    total_weights = 1e-8
    fc_zero_weights = 0.0
    fc_total_weights = 1e-8
    for n, m in params:
        zero_weights += torch.sum(m.weight == 0)
        total_weights += m.weight.nelement()
        if "fc" in n or "6" in n:
            fc_zero_weights += torch.sum(m.weight == 0)
            fc_total_weights += m.weight.nelement()
            
    conv_sparsity = (zero_weights - fc_zero_weights) / (total_weights - fc_total_weights)
    fc_sparsity = fc_zero_weights / fc_total_weights
    global_sparsity = zero_weights / total_weights
    return conv_sparsity, fc_sparsity, global_sparsity