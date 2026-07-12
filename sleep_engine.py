import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

class Adapter(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, action_size=2):
        super(Adapter, self).__init__()
        # Microscopic network for edge execution
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x):
        return self.net(x)

class SleepEngine:
    def __init__(self, working_memory, lambda_time=0.1):
        self.working_memory = working_memory
        self.lambda_time = lambda_time
        
    def score_pairs(self):
        buffer = self.working_memory.memory_buffer
        if len(buffer) < 2:
            return None, None
            
        z_vectors = np.array([m[1] for m in buffer])
        timestamps = np.array([m[0] for m in buffer])
        
        cos_sim = cosine_similarity(z_vectors)
        time_diff = np.abs(timestamps[:, None] - timestamps[None, :])
        
        max_time_diff = np.max(time_diff) if np.max(time_diff) > 0 else 1.0
        p_time = 1.0 - (time_diff / max_time_diff)
        
        # Maximize: Temporal closeness and Semantic distinctness (1 - Cosine Sim)
        score_matrix = (1.0 - cos_sim) + self.lambda_time * p_time
        np.fill_diagonal(score_matrix, -np.inf)
        
        idx_flat = np.argmax(score_matrix)
        i, j = np.unravel_index(idx_flat, score_matrix.shape)
        
        return i, j
        
    def dream_and_train(self, z_new, epochs=50, steps=200):
        adapter = Adapter()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        z_tensor = torch.tensor(z_new, dtype=torch.float32).unsqueeze(0) # (1, latent_size)
        
        # Use the true Generative Decoder to synthesize the concept
        with torch.no_grad():
            state_recon, action_recon = self.working_memory.decode(z_tensor)
            
        # Synthesize a dataset around this concept (adding noise for robustness)
        X = []
        Y = []
        for _ in range(steps):
            noise_x = torch.randn_like(state_recon) * 0.1
            noise_a = torch.randn_like(action_recon) * 0.05
            X.append(state_recon + noise_x)
            Y.append(action_recon + noise_a)
            
        X_train = torch.cat(X, dim=0)
        Y_train = torch.cat(Y, dim=0)
        
        # Train the adapter module on the synthesized dream
        adapter.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = adapter(X_train)
            loss = criterion(preds, Y_train)
            loss.backward()
            optimizer.step()
            
        adapter.eval()
        return adapter
        
    def run_sleep_cycle(self):
        print("Sleep Phase Activated: Scanning memory nodes...")
        i, j = self.score_pairs()
        if i is None:
            print("Not enough memory to sleep.")
            return None, None
            
        z_i = self.working_memory.memory_buffer[i][1]
        z_j = self.working_memory.memory_buffer[j][1]
        
        # Mathematical interpolation
        alpha = 0.5
        z_new = alpha * z_i + (1 - alpha) * z_j
        
        print(f"Synthesizing new concept from memory {i} and {j}...")
        new_adapter = self.dream_and_train(z_new)
        print("Success: Synthesized Adapter!")
        
        return z_new, new_adapter
