import torch


def dr_grpo_loss(
    advantages,
    logprobs,
    eps=1e-8,
):
    """
    Paper-grade Dr.GRPO loss:

        L = - E_group [ A_i * log π(y_i|x) ]

    where:
        advantages: List[Tensor]  (per prompt, shape [group])
        logprobs:   List[Tensor]  (per prompt, shape [group])

    For each prompt P_k:
        L_k = - mean_i [ A_{k,i} * logπ(y_{k,i}|P_k) ]

    Final objective:
        L = mean_k L_k
    """

    losses = []

    for A, lp in zip(advantages, logprobs):
        # ensure shapes match group dimension
        A = A.to(lp.device)

        # group reduce
        # minus sign → gradient ascent on A*logπ
        L_k = - torch.mean(A * lp)
        losses.append(L_k)

    loss = torch.mean(torch.stack(losses))
    return loss
