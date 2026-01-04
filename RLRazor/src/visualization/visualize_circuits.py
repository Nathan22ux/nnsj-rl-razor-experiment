"""
Standalone visualization script for circuit analysis results.

Usage:
    python visualize_circuits.py <path_to_results.json>
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuits.visualization import generate_all_visualizations, print_circuit_summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_circuits.py <results_json_path>")
        print("\nExample:")
        print("  python visualize_circuits.py results/circuits/circuit_analysis_math.json")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    if not os.path.exists(results_path):
        print(f"Error: File not found: {results_path}")
        sys.exit(1)
    
    print(f"\n📊 Generating visualizations for: {results_path}\n")
    
    # Print text summary
    print_circuit_summary(results_path)
    
    # Generate all plots
    generate_all_visualizations(results_path)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()