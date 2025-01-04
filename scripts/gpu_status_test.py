import torch
import sys

try:
    x = torch.zeros(10).cuda()
    print("✅ GPU Test Passed - Created tensor:", x)
    sys.exit(0)
except Exception as e:
    print("❌ GPU Test Failed -", str(e))
    sys.exit(1) 