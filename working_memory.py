import torch
import torch.nn as nn
from ncps.torch import CfC
import time

class WorkingMemory(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, latent_size=16, action_size=2, buffer_capacity=10000):
        super(WorkingMemory, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.action_size = action_size
        
        # Core Liquid Network
        self.cfc = CfC(input_size, hidden_size, batch_first=True)
        
        # Latent Projection
        self.latent_proj = nn.Linear(hidden_size, latent_size)
        
        # Action Projection
        self.action_proj = nn.Linear(latent_size, action_size)
        
        # True Generative Decoder (Reconstructs state and action from z)
        # This is strictly used during the Sleep Phase for synthesis
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, input_size + action_size)
        )
        
        self.memory_buffer = []
        self.buffer_capacity = buffer_capacity
        self.hx = None # hidden state memory
        
    def forward(self, x):
        """
        Forward pass for action selection.
        x: (batch, seq, input_size) or (batch, input_size) or (input_size)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        out, self.hx = self.cfc(x, self.hx)
        
        h_t = out[:, -1, :]
        z_t = self.latent_proj(h_t)
        a_t = self.action_proj(z_t)
        
        return a_t, z_t, h_t
        
    def decode(self, z):
        """
        Decodes a latent vector z into a state-action pair.
        Returns: (state_reconstruction, action_reconstruction)
        """
        recon = self.decoder(z)
        state_recon = recon[:, :self.input_size]
        action_recon = recon[:, self.input_size:]
        return state_recon, action_recon
        
    def reset_hidden(self):
        self.hx = None
        
    def store_experience(self, z_t, state, action, reward, next_state):
        timestamp = time.time()
        
        if isinstance(z_t, torch.Tensor):
            z_val = z_t.detach().cpu().numpy().flatten()
        else:
            z_val = z_t
            
        self.memory_buffer.append((timestamp, z_val))
        
        if len(self.memory_buffer) > self.buffer_capacity:
            self.memory_buffer.pop(0)
