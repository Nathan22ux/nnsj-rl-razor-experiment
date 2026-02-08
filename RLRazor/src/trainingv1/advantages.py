import torch


def compute_group_advantages(
    rewards,
    normalize=True,
    rank_normalize=True,
    eps=1e-8,
):
    """
    Compute Dr.GRPO-style group advantages for binary rewards.

    Input:
        rewards: List[Tensor] with shape [group_size]
                 One tensor per prompt with r_i ∈ {0,1}

    Steps (paper-grade):
    ---------------------------------------------------
    1. baseline:     A_i = r_i − mean(r)
    2. variance norm: A_i = A_i / (std + eps)
    3. rank norm:     A_i = rank(A_i) / N

    Returns:
        advantages: List[Tensor] same shape as rewards
    """

    advantages = []

    for r in rewards:
        # ensure float tensor on device
        r = r.float()

        # step 1: baseline subtraction (reduce bias)
        A = r - r.mean()

        if normalize:
            # step 2: variance normalization (stability)
            std = A.std()
            A = A / (std + eps)

        if rank_normalize:
            # step 3: rank normalization (paper option for sparse rewards)
            # Convert to CPU for argsort if needed
            ranks = torch.argsort(torch.argsort(A))
            ranks = ranks.float() / (len(r) - 1 + eps)
            A = ranks.to(A.device)

        advantages.append(A)

    return advantages
