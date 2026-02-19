import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_pareto_frontier(x_vals, y_vals):
    """
    Compute the Pareto frontier for maximizing both x and y.
    
    For a point to be on the Pareto frontier, there should be no other point
    that is better in both dimensions. We scan from highest x to lowest x,
    keeping track of the maximum y seen so far.
    
    Args:
        x_vals: List of x values (e.g., new task performance)
        y_vals: List of y values (e.g., prior task performance)
        
    Returns:
        Tuple of (pareto_x, pareto_y) containing points on the frontier
    """
    if not x_vals or not y_vals:
        return [], []
    
    # Combine into points
    points = list(zip(x_vals, y_vals))
    
    # Sort by x DESCENDING (highest x first)
    # This way we scan from right to left on the plot
    points.sort(key=lambda p: p[0], reverse=True)
    
    pareto = []
    max_y = -float('inf')
    
    # Iterate from highest x to lowest x
    # A point is on the frontier if its y is >= the best y we've seen
    # from all points with higher x values
    for x, y in points:
        if y >= max_y:
            pareto.append((x, y))
            max_y = y
    
    # Reverse to get ascending x order for plotting (left to right)
    pareto = pareto[::-1]
    
    if pareto:
        pareto_x, pareto_y = zip(*pareto)
        return list(pareto_x), list(pareto_y)
    return [], []


def get_near_pareto_points(x_vals, y_vals, threshold=2.0):
    """
    Get points within `threshold` of the Pareto frontier (paper's methodology).
    
    From the paper (Appendix B.1):
    "From the trained models, we retained only those lying within 2 accuracy 
    points of the Pareto frontier."
    
    Args:
        x_vals: List of x values
        y_vals: List of y values  
        threshold: Distance threshold from frontier (default 2.0 accuracy points)
        
    Returns:
        Tuple of (near_x, near_y) containing points near the frontier
    """
    if not x_vals or not y_vals:
        return [], []
    
    points = list(zip(x_vals, y_vals))
    pareto_x, pareto_y = compute_pareto_frontier(x_vals, y_vals)
    
    if not pareto_x:
        return [], []
    
    # For each point, compute distance to Pareto frontier
    # Distance = how much better could y be for this x?
    near_pareto = []
    
    for x, y in points:
        # Find the best y on the frontier for points with x >= this x
        best_y_for_x = -float('inf')
        for px, py in zip(pareto_x, pareto_y):
            if px >= x:
                best_y_for_x = max(best_y_for_x, py)
        
        # If no frontier point has higher x, use the rightmost frontier point
        if best_y_for_x == -float('inf'):
            best_y_for_x = pareto_y[-1]  # rightmost point on frontier
        
        # Check if within threshold
        distance = best_y_for_x - y
        if distance <= threshold:
            near_pareto.append((x, y))
    
    if near_pareto:
        near_x, near_y = zip(*near_pareto)
        return list(near_x), list(near_y)
    return [], []


def fit_frontier_curve(x_vals, y_vals, num_points=100, base_model_pt=None, extend_left_to=None):
    """
    Fit an exponential decay curve to points near the Pareto frontier.
    
    From the paper (Appendix B.1):
    "An exponential function was fit to the filtered points to produce 
    the trade-off curves."
    
    The paper's curves start flat on the left (base model has low NT but high PT)
    and then curve downward as NT increases (forgetting occurs).
    
    Args:
        x_vals: List of x values (points near frontier)
        y_vals: List of y values (points near frontier)
        num_points: Number of points for the smooth curve
        base_model_pt: Prior task score of base model (to set the flat plateau)
        extend_left_to: Extend curve leftward to this x value (for flat start)
        
    Returns:
        Tuple of (curve_x, curve_y) for plotting
    """
    from scipy.optimize import curve_fit
    
    if len(x_vals) < 3:
        return list(x_vals), list(y_vals)
    
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    
    # Sort by x for fitting
    sort_idx = np.argsort(x_arr)
    x_arr = x_arr[sort_idx]
    y_arr = y_arr[sort_idx]
    
    # Determine the plateau value (base model PT or max PT in data)
    if base_model_pt is not None:
        plateau = base_model_pt
    else:
        plateau = np.max(y_arr)
    
    # Exponential decay from plateau: y = plateau - a * exp(b * (x - c))
    # This creates a flat region on the left that curves down on the right
    def exp_decay(x, a, b, c):
        return plateau - a * np.exp(b * (x - c))
    
    # Alternative: logistic decay (smoother transition)
    def logistic_decay(x, k, x0, L):
        return plateau - L / (1 + np.exp(-k * (x - x0)))
    
    # Determine x range for curve
    x_min = extend_left_to if extend_left_to is not None else x_arr.min()
    x_max = x_arr.max()
    curve_x = np.linspace(x_min, x_max, num_points)
    
    # Try fitting
    curve_y = None
    
    # Method 1: Exponential decay
    try:
        # Initial guess: small a, positive b, c near the transition point
        p0 = [0.1, 0.05, np.median(x_arr)]
        bounds = ([0, 0, x_min], [10, 1, x_max])
        popt, _ = curve_fit(exp_decay, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=5000)
        curve_y = exp_decay(curve_x, *popt)
        # Clip to not exceed plateau
        curve_y = np.clip(curve_y, None, plateau)
    except:
        pass
    
    # Method 2: Logistic decay (if exponential fails)
    if curve_y is None:
        try:
            p0 = [0.1, np.median(x_arr), 1.0]
            bounds = ([0.001, x_min, 0.01], [1, x_max, 10])
            popt, _ = curve_fit(logistic_decay, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=5000)
            curve_y = logistic_decay(curve_x, *popt)
            curve_y = np.clip(curve_y, None, plateau)
        except:
            pass
    
    # Method 3: Polynomial fit as fallback
    if curve_y is None:
        try:
            # Use quadratic fit but constrain to not exceed plateau
            coeffs = np.polyfit(x_arr, y_arr, 2)
            poly = np.poly1d(coeffs)
            curve_y = poly(curve_x)
            curve_y = np.clip(curve_y, None, plateau)
        except:
            # Last resort: linear interpolation with flat extension
            curve_y = np.interp(curve_x, x_arr, y_arr)
            # Extend flat to the left
            curve_y[curve_x < x_arr.min()] = plateau
    
    return list(curve_x), list(curve_y)


def plot_pareto_frontier(results, dataset_name="math", use_curve_fit=True, threshold=2.0,
                         base_model_nt=None, base_model_pt=None, extend_left=True):
    """
    Create the Pareto frontier plot (Figure 2 from the paper).
    Shows New Task Performance vs Prior Task Performance trade-off.
    
    Follows the paper's methodology (Appendix B.1):
    1. Filter points within `threshold` of the Pareto frontier
    2. Fit an exponential curve to the filtered points
    3. Extend curve leftward to show flat region (base model performance)
    
    Args:
        results: Dictionary containing 'sft' and 'rl' results
        dataset_name: Name of the dataset (for title)
        use_curve_fit: If True, fit curve like the paper. If False, just connect points.
        threshold: Distance threshold for near-Pareto filtering (default 2.0)
        base_model_nt: New task accuracy of base model before fine-tuning (optional)
        base_model_pt: Prior task score of base model before fine-tuning (optional)
        extend_left: If True, extend curve leftward to show flat region
    """
    print("\n" + "="*70)
    print("CREATING PARETO FRONTIER PLOT (Figure 2)")
    print("="*70)
    
    # Extract data (skip missing PT from older result files)
    print("\n Extracting data from results...")
    sft_points = [(r.get('NT'), r.get('PT')) for r in results.get('sft', [])]
    rl_points = [(r.get('NT'), r.get('PT')) for r in results.get('rl', [])]

    sft_points = [(nt, pt) for nt, pt in sft_points if nt is not None and pt is not None]
    rl_points = [(nt, pt) for nt, pt in rl_points if nt is not None and pt is not None]

    sft_nt = [nt for nt, _ in sft_points]
    sft_pt = [pt for _, pt in sft_points]
    rl_nt = [nt for nt, _ in rl_points]
    rl_pt = [pt for _, pt in rl_points]

    if not sft_points and not rl_points:
        print("Warning: No PT values found in results; skipping Pareto frontier plot.")
        return
    
    print(f"Extracted {len(sft_nt)} SFT results and {len(rl_nt)} RL results")
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Plot all points
    if sft_nt:
        plt.scatter(sft_nt, sft_pt, label='SFT', alpha=0.5, s=80, color='blue', marker='o')
    if rl_nt:
        plt.scatter(rl_nt, rl_pt, label='RL (GRPO)', alpha=0.5, s=80, color='red', marker='o')
    
    # Determine extend_left_to value
    all_nt = sft_nt + rl_nt
    if extend_left and all_nt:
        # Extend to ~80% of minimum x, or to base_model_nt if provided
        if base_model_nt is not None:
            extend_left_to = base_model_nt
        else:
            extend_left_to = min(all_nt) * 0.7  # Extend 30% to the left
    else:
        extend_left_to = None
    
    # Get points near Pareto frontier and fit curves (paper methodology)
    if use_curve_fit:
        # SFT frontier
        sft_near_x, sft_near_y = get_near_pareto_points(sft_nt, sft_pt, threshold=threshold)
        print(f"SFT points near frontier (within {threshold}): {len(sft_near_x)}")
        
        if len(sft_near_x) >= 3:
            sft_curve_x, sft_curve_y = fit_frontier_curve(
                sft_near_x, sft_near_y, 
                base_model_pt=base_model_pt,
                extend_left_to=extend_left_to
            )
            plt.plot(sft_curve_x, sft_curve_y, '--', color='blue', 
                    linewidth=2.5, label='SFT Frontier', alpha=0.9)
        elif len(sft_near_x) >= 2:
            # Just connect the points if not enough for curve fit
            sorted_pts = sorted(zip(sft_near_x, sft_near_y))
            px, py = zip(*sorted_pts)
            plt.plot(px, py, 'o--', color='blue', linewidth=2.5, 
                    markersize=8, label='SFT Frontier', alpha=0.8)
        
        # RL frontier  
        rl_near_x, rl_near_y = get_near_pareto_points(rl_nt, rl_pt, threshold=threshold)
        print(f"RL points near frontier (within {threshold}): {len(rl_near_x)}")
        
        if len(rl_near_x) >= 3:
            rl_curve_x, rl_curve_y = fit_frontier_curve(
                rl_near_x, rl_near_y,
                base_model_pt=base_model_pt,
                extend_left_to=extend_left_to
            )
            plt.plot(rl_curve_x, rl_curve_y, '--', color='red', 
                    linewidth=2.5, label='RL Frontier', alpha=0.9)
        elif len(rl_near_x) >= 2:
            sorted_pts = sorted(zip(rl_near_x, rl_near_y))
            px, py = zip(*sorted_pts)
            plt.plot(px, py, 's--', color='red', linewidth=2.5,
                    markersize=8, label='RL Frontier', alpha=0.8)
    else:
        # Original method: just connect Pareto frontier points
        sft_pareto_x, sft_pareto_y = compute_pareto_frontier(sft_nt, sft_pt)
        rl_pareto_x, rl_pareto_y = compute_pareto_frontier(rl_nt, rl_pt)
        
        print(f"SFT Pareto frontier: {len(sft_pareto_x)} points")
        print(f"RL Pareto frontier: {len(rl_pareto_x)} points")
        
        if len(sft_pareto_x) > 1:
            plt.plot(sft_pareto_x, sft_pareto_y, 'o--', color='orange', 
                    linewidth=2.5, markersize=8, label='SFT Frontier', alpha=0.8)
        elif len(sft_pareto_x) == 1:
            plt.scatter(sft_pareto_x, sft_pareto_y, color='orange', s=150, 
                       marker='o', edgecolors='black', linewidths=2, label='SFT Frontier')
        
        if len(rl_pareto_x) > 1:
            plt.plot(rl_pareto_x, rl_pareto_y, 's--', color='blue', 
                    linewidth=2.5, markersize=8, label='RL Frontier', alpha=0.8)
        elif len(rl_pareto_x) == 1:
            plt.scatter(rl_pareto_x, rl_pareto_y, color='blue', s=150,
                       marker='s', edgecolors='black', linewidths=2, label='RL Frontier')
    
    # Labels and title
    plt.xlabel('New Task Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Avg. Score on Previous Tasks', fontsize=14, fontweight='bold')
    plt.title(f'Pareto Frontier: SFT vs RL ({dataset_name.capitalize()})', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    filename = f'pareto_frontier_{dataset_name}.png'
    os.makedirs("results", exist_ok=True)
    plt.savefig(f'results/{filename}', dpi=200, bbox_inches='tight')
    print(f"Saved: results/{filename}")
    plt.show()
    
    print("\n" + "="*70)
    print("PARETO FRONTIER PLOT COMPLETE")
    print("="*70)


def plot_results(results):
    """
    Create visualizations for the experiment results.
    Generates two plots:
    1. KL vs Prior Task Performance (showing forgetting)
    2. SFT vs RL Comparison
    
    Args:
        results: Dictionary containing 'sft' and 'rl' results
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Extract data
    print("\n Extracting data from results...")
    sft_kl = [r.get('kl_divergence') for r in results.get('sft', []) if r.get('kl_divergence') is not None]
    rl_kl = [r.get('kl_divergence') for r in results.get('rl', []) if r.get('kl_divergence') is not None]

    sft_prior = [r.get('PT') for r in results.get('sft', []) if r.get('PT') is not None]
    rl_prior = [r.get('PT') for r in results.get('rl', []) if r.get('PT') is not None]

    print(f"Extracted {len(sft_kl)} SFT KL results and {len(rl_kl)} RL KL results")
    
    # KL vs Prior Task (showing forgetting) - only if PT exists
    if sft_prior or rl_prior:
        sft_pairs = [(r.get('kl_divergence'), r.get('PT')) for r in results.get('sft', [])]
        rl_pairs = [(r.get('kl_divergence'), r.get('PT')) for r in results.get('rl', [])]
        sft_pairs = [(k, p) for k, p in sft_pairs if k is not None and p is not None]
        rl_pairs = [(k, p) for k, p in rl_pairs if k is not None and p is not None]

        if sft_pairs or rl_pairs:
            sft_kl_plot = [k for k, _ in sft_pairs]
            sft_prior_plot = [p for _, p in sft_pairs]
            rl_kl_plot = [k for k, _ in rl_pairs]
            rl_prior_plot = [p for _, p in rl_pairs]

            print("\n Creating Plot 1: KL vs Prior Task Performance...")
            plt.figure(figsize=(10, 6))
            if sft_kl_plot:
                plt.scatter(sft_kl_plot, sft_prior_plot, label='SFT', alpha=0.6, s=50, color='orange')
            if rl_kl_plot:
                plt.scatter(rl_kl_plot, rl_prior_plot, label='RL', alpha=0.6, s=50, color='blue')
            plt.xlabel('KL Divergence', fontsize=12)
            plt.ylabel('Prior Task Performance (%)', fontsize=12)
            plt.title('KL Predicts Forgetting (Lower KL = Less Forgetting)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            os.makedirs("results", exist_ok=True)
            plt.savefig('results/kl_vs_forgetting.png', dpi=150)
            print("Saved: results/kl_vs_forgetting.png")
            plt.show()
        else:
            print("\nWarning: Skipping Plot 1 (no aligned KL/PT pairs).")
    else:
        print("\nWarning: Skipping Plot 1 (PT missing in results).")
    
    # Comparison plot
    print("\n Creating Plot 2: SFT vs RL Comparison...")
    
    methods = ['SFT', 'RL']
    prior_scores = [
        np.mean(sft_prior) if sft_prior else 0.0,
        np.mean(rl_prior) if rl_prior else 0.0
    ]
    kl_divs = [
        np.mean(sft_kl) if sft_kl else 0.0,
        np.mean(rl_kl) if rl_kl else 0.0
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, prior_scores, width, label='Prior Task Score', alpha=0.8, color=['orange', 'blue'])
    ax.bar(x + width/2, kl_divs, width, label='KL Divergence', alpha=0.8, color=['#ffcc80', '#80b3ff'])
    
    ax.set_ylabel('Score')
    ax.set_title('SFT vs RL: Forgetting Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/sft_vs_rl_comparison.png', dpi=150)
    print("Saved: results/sft_vs_rl_comparison.png")
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


def plot_NT_PT(results):
    """
    NEW TASK vs PRIOR TASK scatter plot (without Pareto frontier lines)
    """
    print("\n" + "="*70)
    print("CREATING NT vs PT PLOT")
    print("="*70)

    sft_points = [(r.get("NT"), r.get("PT")) for r in results.get("sft", [])]
    rl_points = [(r.get("NT"), r.get("PT")) for r in results.get("rl", [])]

    sft_points = [(nt, pt) for nt, pt in sft_points if nt is not None and pt is not None]
    rl_points = [(nt, pt) for nt, pt in rl_points if nt is not None and pt is not None]

    if not sft_points and not rl_points:
        print("Warning: No PT values found in results; skipping NT vs PT plot.")
        return

    sft_nt = [nt for nt, _ in sft_points]
    sft_pt = [pt for _, pt in sft_points]
    rl_nt = [nt for nt, _ in rl_points]
    rl_pt = [pt for _, pt in rl_points]

    print(f"Plotting {len(sft_nt)} SFT points and {len(rl_nt)} RL points")

    plt.figure(figsize=(10, 8))
    if sft_nt:
        plt.scatter(sft_nt, sft_pt, color="orange", label="SFT", alpha=0.7, s=40)
    if rl_nt:
        plt.scatter(rl_nt, rl_pt, color="blue", label="RL (GRPO)", alpha=0.7, s=40)

    plt.xlabel("New Task Performance (NT)", fontsize=12)
    plt.ylabel("Prior Tasks Performance (PT)", fontsize=12)
    plt.title("NT vs PT Comparison (SFT vs RL)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/nt_vs_pt.png", dpi=300)
    print("Saved: results/nt_vs_pt.png")
    plt.show()
    
    print("\n" + "="*70)
    print("NT vs PT PLOT COMPLETE")
    print("="*70)


def test_pareto_frontier():
    """
    Test the Pareto frontier computation with known examples.
    """
    print("\n" + "="*70)
    print("TESTING PARETO FRONTIER COMPUTATION")
    print("="*70)
    
    # Test case: points where frontier should be top-right envelope
    x = [1, 2, 3, 4, 5, 2, 3]
    y = [5, 3, 4, 2, 3, 4, 3]
    
    pareto_x, pareto_y = compute_pareto_frontier(x, y)
    
    print(f"Input points: {list(zip(x, y))}")
    print(f"Pareto frontier: {list(zip(pareto_x, pareto_y))}")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=100, alpha=0.6, label='All points')
    plt.plot(pareto_x, pareto_y, 'ro-', markersize=12, linewidth=2, label='Pareto frontier')
    plt.xlabel('X (New Task)')
    plt.ylabel('Y (Prior Task)')
    plt.title('Pareto Frontier Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/pareto_test.png", dpi=150)
    print("Saved: results/pareto_test.png")
    plt.show()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_pareto_frontier()