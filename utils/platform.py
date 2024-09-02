import torch
import platform

def get_accelerator():
    if platform.system() == "Darwin":
        try:
            import torch
            import torch.mps
            return torch.device("mps")
        except ImportError:
            pass

    try:
        import torch.cuda
        if torch.cuda.is_available():
            return torch.device("mps")
    except ImportError:
        pass

    return torch.device("cpu")