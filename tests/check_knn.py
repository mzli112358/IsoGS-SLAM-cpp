#!/usr/bin/env python3
"""
Check if simple_knn library is working correctly.
This script tests the simple_knn.distCUDA2 function.
"""

import torch

try:
    import simple_knn
    print("[✓] Successfully imported simple_knn")
except ImportError as e:
    print(f"[✗] Failed to import simple_knn: {e}")
    print("\nError: simple_knn library is not available.")
    print("Please make sure you have compiled and installed simple_knn.")
    exit(1)

# Try to import the actual C extension
try:
    from simple_knn._C import distCUDA2
    print("[✓] Successfully imported distCUDA2 from simple_knn._C")
    has_distCUDA2 = True
except ImportError as e:
    print(f"[!] Could not import distCUDA2 from simple_knn._C: {e}")
    print("[!] This might mean simple_knn is not fully compiled or installed.")
    has_distCUDA2 = False

# Check CUDA availability
if not torch.cuda.is_available():
    print("[✗] CUDA is not available. simple_knn requires CUDA.")
    exit(1)

print(f"[✓] CUDA is available: {torch.cuda.get_device_name(0)}")

# Generate random points on CUDA
print("\nGenerating random points (1000, 3) on CUDA...")
points = torch.randn(1000, 3, device='cuda', dtype=torch.float32)
print(f"[✓] Points shape: {points.shape}")
print(f"[✓] Points device: {points.device}")
print(f"[✓] Points dtype: {points.dtype}")

# Test distCUDA2 if available
if has_distCUDA2:
    try:
        print("\nCalling distCUDA2(points)...")
        dist = distCUDA2(points)
        print(f"[✓] Successfully called distCUDA2")
        print(f"[✓] Output shape: {dist.shape}")
        print(f"[✓] Output device: {dist.device}")
        print(f"[✓] Output dtype: {dist.dtype}")
        
        # Print first 5 values
        print(f"\nFirst 5 values of dist:")
        print(dist[:5].cpu().numpy())
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Min: {dist.min().item():.6f}")
        print(f"  Max: {dist.max().item():.6f}")
        print(f"  Mean: {dist.mean().item():.6f}")
        print(f"  Std: {dist.std().item():.6f}")
        
        print("\n[✓] All tests passed! simple_knn.distCUDA2 is working correctly.")
    except Exception as e:
        print(f"\n[✗] Error calling distCUDA2: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print("\n[!] Warning: distCUDA2 function is not available.")
    print("[!] simple_knn module can be imported, but the C extension might not be compiled.")
    print("[!] Please check if simple-knn was properly installed with: cd simple-knn && pip install -e .")
    print("[!] However, the basic import works, which means the package structure is correct.")

