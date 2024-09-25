import torch
import torch.nn as nn

class GradientMapping(nn.Module):
    def __init__(self): 
        super().__init__()
        
        self.sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        self.sobel_x = self.sobel_x.repeat(3, 1, 1, 1)
        self.sobel_y = self.sobel_y.repeat(3, 1, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates the summation of Gradient Maps over all channels.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W).
        """
        B, C, H, W = x.shape
        
        Ix = torch.conv2d(x, self.sobel_x, padding=1, groups=C)
        Iy = torch.conv2d(x, self.sobel_y, padding=1, groups=C)

        G = torch.hypot(Ix, Iy)
        G = torch.sum(G, dim=1)
        return G

class SourceEdgeMask(nn.Module): 
    def __init__(self, e_min : float = 0.1):
        super().__init__()
        
        self.e_min = e_min
        
        self.g_map = GradientMapping()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Computes the Source Edge Mask from the Gradient map of a given image. 
        
        Args: 
            x (torch.Tensor): input tensor of shape (B, C, H, W)
        
        Returns: 
            torch.Tensor: Source Edge Mask tensor of shape (B, H, W)
        """
        B, C, H, W = x.shape
        G_i = self.g_map(x)
        max_G_i = torch.max(G_i.view(B, H*W)).item()
        
        E_s = torch.min(G_i/(self.e_min * max_G_i), torch.ones((B, H, W)))
        
        return E_s