from logger import get_logger

logger = get_logger("eval_pareto")


def dominates(a, b, x_key, y_key):
    """
    Returns True if point a dominates point b.

    For RL's Razor:
        y-axis (NT): higher is better
        x-axis (KL or Forgetting): lower is better
    """
    return (
        a[y_key] >= b[y_key] and
        a[x_key] <= b[x_key] and
        (a[y_key] > b[y_key] or a[x_key] < b[x_key])
    )


def pareto_frontier(points, x_key, y_key):
    """
    Compute Pareto frontier for RL's Razor plots.

    Args:
        x_key: "KL" or "forgetting"
        y_key: "NT"

    Returns:
        List of non-dominated points
    """

    logger.info("=" * 60)
    logger.info(f"Computing Pareto frontier for x={x_key}, y={y_key}")
    logger.info("=" * 60)

    pareto = []

    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p, x_key, y_key):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    # sort for plotting convenience
    pareto = sorted(pareto, key=lambda d: d[x_key])

    logger.info(f"Pareto frontier size: {len(pareto)}")
    return pareto


def pareto_kl_nt(points):
    """
    Convenience wrapper for KL vs NT plot.
    """
    return pareto_frontier(points, x_key="KL", y_key="NT")


def pareto_forgetting_nt(points):
    """
    Convenience wrapper for Forgetting vs NT plot.
    """
    return pareto_frontier(points, x_key="forgetting", y_key="NT")
