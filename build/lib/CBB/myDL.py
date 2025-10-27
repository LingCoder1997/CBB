import torch
import torchvision
import torch.nn as nn
from torchviz import make_dot

def model_is_torch(model):
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a PyTorch model (nn.Module), but got {type(model).__name__}")

def visualize_model(model, dummy_input, output_file="model_graph"):
    import torch
    import torch.nn as nn
    from torchviz import make_dot

    if isinstance(model, nn.DataParallel):
        model = model.module

    if not isinstance(model, nn.Module):
        raise TypeError("Expected a PyTorch nn.Module model.")
    if not isinstance(dummy_input, torch.Tensor):
        raise TypeError("Expected dummy_input to be a torch.Tensor.")

    # 设备对齐
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)

    try:
        output = model(dummy_input)
    except Exception as e:
        raise ValueError(f"Model forward failed. Check input shape and device.\nDetails: {e}")

    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(output_file, format="png")
    print(f"✅ 模型图保存为 {output_file}.png")