import torch
import time

def test_gpu_operations():
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Basic tensor operations
    print("\n=== Test 1: Basic Tensor Operations ===")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    start_time = time.time()
    c = torch.matmul(a, b)
    end_time = time.time()
    
    print(f"Matrix multiplication shape: {c.shape}")
    print(f"Operation time: {(end_time - start_time)*1000:.2f} ms")
    
    # Test 2: Memory transfer
    print("\n=== Test 2: Memory Transfer ===")
    start_time = time.time()
    cpu_tensor = torch.randn(10000, 10000)
    gpu_tensor = cpu_tensor.to(device)
    end_time = time.time()
    
    print(f"Memory transfer time: {(end_time - start_time)*1000:.2f} ms")
    
    # Test 3: Element-wise operations
    print("\n=== Test 3: Element-wise Operations ===")
    x = torch.randn(5000, 5000, device=device)
    
    start_time = time.time()
    y = torch.sin(x) + torch.cos(x)
    torch.cuda.synchronize()  # Make sure operation is complete
    end_time = time.time()
    
    print(f"Element-wise operation time: {(end_time - start_time)*1000:.2f} ms")
    
    return "All GPU tests completed successfully!"

if __name__ == "__main__":
    try:
        result = test_gpu_operations()
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 