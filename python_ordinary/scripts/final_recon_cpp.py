"""
Final reconstruction visualization using C++ backend
"""
import numpy as np
import torch
import sys
import os

# Try to import C++ bindings
try:
    import isogs_viz
    HAS_CPP_VIZ = True
except ImportError:
    HAS_CPP_VIZ = False
    print("Warning: C++ visualization bindings not available, using Python fallback")

def visualize_final_recon(config_path, checkpoint_path=None):
    """
    Visualize final SLAM reconstruction using C++ backend
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file (if None, auto-detect latest)
    """
    if HAS_CPP_VIZ:
        # Use C++ visualizer
        viz = isogs_viz.FinalVisualizer()
        viz.initialize(1280, 720)
        
        if checkpoint_path is None:
            # Auto-detect latest checkpoint
            # TODO: Implement auto-detection
            checkpoint_path = "checkpoints/params.npz"
        
        viz.load_and_show(checkpoint_path)
        viz.run()
        viz.close()
    else:
        # Fallback to Python visualization
        print("Using Python visualization fallback")
        # TODO: Implement Python fallback

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_recon_cpp.py <config_path> [checkpoint_path]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_final_recon(config_path, checkpoint_path)

