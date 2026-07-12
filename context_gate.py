import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextGate(nn.Module):
    def __init__(self, latent_size=16):
        super(ContextGate, self).__init__()
        self.latent_size = latent_size
        self.keys = [] # list of z_k (tensors)
        self.adapters = [] # list of Adapter modules
        
    def add_adapter(self, z_k, adapter):
        if not isinstance(z_k, torch.Tensor):
            z_k = torch.tensor(z_k, dtype=torch.float32)
        self.keys.append(z_k)
        self.adapters.append(adapter)
        
    def forward(self, a_base, x_t, q_z):
        """
        a_base: (batch, action_size)
        x_t: (batch, state_size)
        q_z: (batch, latent_size)
        """
        if len(self.adapters) == 0:
            return a_base
            
        keys_tensor = torch.stack(self.keys) # (M, latent_size)
        
        # Soft-attention calculation
        scores = torch.matmul(q_z, keys_tensor.T) # (batch, M)
        alphas = F.softmax(scores, dim=-1)
        
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_outputs.append(adapter(x_t))
            
        adapter_outputs_tensor = torch.stack(adapter_outputs) # (M, batch, action_size)
        
        alphas_expanded = alphas.T.unsqueeze(-1) # (M, batch, 1)
        blended_adapter_action = torch.sum(alphas_expanded * adapter_outputs_tensor, dim=0)
        
        a_final = a_base + blended_adapter_action
        return a_final
