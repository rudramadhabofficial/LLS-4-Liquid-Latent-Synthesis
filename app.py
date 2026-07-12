import gradio as gr
import plotly.graph_objects as go
import numpy as np
import time
import torch
from sklearn.decomposition import PCA

from environment import Environment
from working_memory import WorkingMemory
from sleep_engine import SleepEngine
from context_gate import ContextGate

# --- Global Controller ---
class LLSController:
    def __init__(self):
        self.env = Environment()
        self.brain = WorkingMemory()
        self.sleep_engine = SleepEngine(self.brain)
        self.gate = ContextGate()
        
        self.state = "AWAKE" 
        self.log_messages = ["[" + time.strftime("%H:%M:%S") + "] SYSTEM ONLINE. Base Continuous-Time Liquid Network Activated."]
        self.step_count = 0
        self.agent_path = []
        self.sleep_cycle_running = False
        self.sleep_requested = False
        
        self.brain.reset_hidden()
        self.current_obs = self.env.reset()
        self.agent_path.append(self.env.agent_pos.copy())
        
    def step_environment(self):
        if self.state != "AWAKE" or self.sleep_cycle_running:
            return
            
        self.step_count += 1
        obs_tensor = torch.tensor(self.current_obs, dtype=torch.float32)
        
        with torch.no_grad():
            a_base, z_t, h_t = self.brain(obs_tensor)
            a_final = self.gate(a_base, obs_tensor.unsqueeze(0), z_t).squeeze(0)
            
            action_np = a_final.numpy()
            next_obs, reward, done, _ = self.env.step(action_np)
            self.brain.store_experience(z_t, self.current_obs, action_np, reward, next_obs)
            
            self.current_obs = next_obs
            self.agent_path.append(self.env.agent_pos.copy())
            
            if len(self.agent_path) > 100:
                self.agent_path.pop(0) # Trail effect
            
            if done:
                self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] Episode Terminated. Energy Resets.")
                self.current_obs = self.env.reset()
                self.brain.reset_hidden()
                self.agent_path = [self.env.agent_pos.copy()]
                
            if reward > 5.0:
                 self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] Target Objective Acquired! +10 Reward.")
                 
            # Keep logs clean
            if len(self.log_messages) > 15:
                self.log_messages.pop(0)

    def request_sleep(self):
        if not self.sleep_cycle_running:
            self.sleep_requested = True

    def run_sleep_sequence(self):
        self.sleep_cycle_running = True
        self.sleep_requested = False
        self.state = "SLEEPING"
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] SLEEP PHASE TRIGGERED. Suspending Motor Functions...")
        yield self.get_ui_update()
        time.sleep(1.0)
        
        mem_count = len(self.brain.memory_buffer)
        if mem_count < 5:
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] Error: Insufficient memory ({mem_count}/5 nodes). Waking up.")
            self.state = "AWAKE"
            self.sleep_cycle_running = False
            yield self.get_ui_update()
            return
            
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] SYNTHESIS: Scanning {mem_count} high-dimensional memory nodes...")
        yield self.get_ui_update()
        time.sleep(1.0)
        
        # Actual math execution
        i, j = self.sleep_engine.score_pairs()
        if i is None:
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] SYNTHESIS ABORTED: No distinct co-occurring patterns found.")
        else:
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] ALGORITHM: High Cosine-Distance + Temporal Proximity match found at indices {i}, {j}.")
            yield self.get_ui_update()
            time.sleep(1.0)
            
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] DREAMING: Decoding interpolated latent z_new into synthetic reality tensors...")
            yield self.get_ui_update()
            time.sleep(1.0)
            
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] TRAINING: Compiling ultra-lightweight Edge Adapter (< 1000 parameters)...")
            yield self.get_ui_update()
            
            # Heavy computation
            z_i = self.brain.memory_buffer[i][1]
            z_j = self.brain.memory_buffer[j][1]
            z_new = 0.5 * z_i + 0.5 * z_j
            adapter = self.sleep_engine.dream_and_train(z_new, epochs=30, steps=100)
            
            self.gate.add_adapter(z_new, adapter)
            self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] SUCCESS: New concept mastered! Adapter #{len(self.gate.adapters)} integrated into Context Gate.")
            
        time.sleep(1.0)
        self.state = "AWAKE"
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] SYSTEM WAKING UP. Resuming continuous physics...")
        self.sleep_cycle_running = False
        yield self.get_ui_update()

    def get_ui_update(self):
        log_text = "\n".join(self.log_messages)
        
        info_html = f"""
        <div style="padding: 10px; background-color: #1e1e1e; color: #00ff00; font-family: monospace; border-radius: 5px; border: 1px solid #333;">
            <h3 style="color: #00ffff; margin-top: 0;">🧬 LLS-4 Neural Telemetry</h3>
            <b>System State:</b> <span style="color: {'#ff0000' if self.state == 'SLEEPING' else '#00ff00'};">{self.state}</span><br>
            <b>Time Steps Survived:</b> {self.step_count}<br>
            <b>Spatial Memories Encoded:</b> {len(self.brain.memory_buffer)}<br>
            <b>Synthetic Adapters Active:</b> {len(self.gate.adapters)}<br>
            <hr style="border-color: #333;">
            <i style="color: #aaaaaa; font-size: 0.9em;">
            <b>Awake Phase:</b> Agent continuously processes 4D physics through a Liquid Neural Network (CfC). Memories are compressed via Deep Autoencoder to a 16D latent manifold.<br><br>
            <b>Sleep Phase:</b> Inference halts. The Engine finds two distinct memories (low cosine similarity) that happened closely in time. It interpolates them mathematically, decodes the new vector into "dream" data, and trains a microscopic neural adapter.
            </i>
        </div>
        """
        
        return (
            info_html, 
            self.plot_trajectory(), 
            self.plot_latent(), 
            log_text
        )
        
    def plot_trajectory(self):
        fig = go.Figure()
        
        # Target
        fig.add_trace(go.Scatter(x=[self.env.target_pos[0]], y=[self.env.target_pos[1]],
                                 mode='markers', marker=dict(color='gold', size=20, symbol='star'), name='Target'))
        # Obstacle
        fig.add_trace(go.Scatter(x=[self.env.obstacle_pos[0]], y=[self.env.obstacle_pos[1]],
                                 mode='markers', marker=dict(color='red', size=15, symbol='square'), name='Obstacle'))
        
        # Path trail
        if len(self.agent_path) > 0:
            path = np.array(self.agent_path)
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', 
                                     line=dict(color='cyan', width=2, dash='dot'), opacity=0.5, name='Memory Trail'))
            # Agent head
            fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode='markers',
                                     marker=dict(color='cyan', size=12, line=dict(color='white', width=2)), name='LLS Agent'))
            
        fig.update_layout(
            title="Live Environmental Physics (2D)",
            plot_bgcolor='rgb(10,10,20)',
            paper_bgcolor='rgb(10,10,20)',
            font=dict(color='white'),
            xaxis=dict(range=[0, 20], showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 20], showgrid=False, zeroline=False),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        return fig
        
    def plot_latent(self):
        fig = go.Figure()
        
        if len(self.brain.memory_buffer) >= 3:
            z_vectors = np.array([m[1] for m in self.brain.memory_buffer])
            
            # Use PCA to project 16D to 3D!
            pca = PCA(n_components=3)
            # If we don't have enough data for 3D, just pad with zeros
            if z_vectors.shape[0] < 3:
                 z_3d = np.pad(z_vectors, ((0,0), (0, 3-z_vectors.shape[1])))
            else:
                 z_3d = pca.fit_transform(z_vectors)
                 
            fig.add_trace(go.Scatter3d(
                x=z_3d[:, 0], y=z_3d[:, 1], z=z_3d[:, 2],
                mode='markers',
                marker=dict(size=4, color='rgba(150, 150, 255, 0.5)'),
                name='Raw Encoded Memories'
            ))
            
            if len(self.gate.keys) > 0:
                keys_tensor = torch.stack(self.gate.keys).numpy()
                try:
                    keys_3d = pca.transform(keys_tensor)
                    fig.add_trace(go.Scatter3d(
                        x=keys_3d[:, 0], y=keys_3d[:, 1], z=keys_3d[:, 2],
                        mode='markers',
                        marker=dict(size=10, color='magenta', symbol='diamond', line=dict(color='white', width=2)),
                        name='Synthesized Concepts'
                    ))
                except:
                    pass
                    
        fig.update_layout(
            title="Latent Space Brain Map (PCA 16D ➔ 3D)",
            plot_bgcolor='rgb(10,10,20)',
            paper_bgcolor='rgb(10,10,20)',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgb(10,10,20)'
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

controller = LLSController()

def run_loop():
    while True:
        if getattr(controller, 'sleep_requested', False) and not controller.sleep_cycle_running:
            for update in controller.run_sleep_sequence():
                yield update
        elif not controller.sleep_cycle_running:
            for _ in range(5):
                controller.step_environment()
            yield controller.get_ui_update()
        time.sleep(0.1)

# --- UI Setup ---
css = """
body { background-color: #0d0d0d !important; color: white !important; }
.gradio-container { max-width: 1400px !important; }
button { background-color: #00ffff !important; color: black !important; font-weight: bold !important; box-shadow: 0 0 10px #00ffff; border: none !important;}
button:hover { background-color: #ffffff !important; box-shadow: 0 0 20px #00ffff; }
textarea { background-color: #000000 !important; color: #00ff00 !important; border: 1px solid #333 !important; font-family: monospace !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; color: #00ffff; text-shadow: 0 0 10px #00ffff;'>🧠 LIQUID LATENT SYNTHESIS (LLS-4) COMMAND CENTER</h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            info_panel = gr.HTML(value="")
            sleep_btn = gr.Button("TRIGGER SYSTEM SLEEP CYCLE", size="lg")
            log_panel = gr.Textbox(label="Mainframe Console Terminal", lines=12, interactive=False)
            
        with gr.Column(scale=2):
            with gr.Row():
                traj_plot = gr.Plot(label="Live Trajectory Map")
                latent_plot = gr.Plot(label="Latent Space Manifold Graph")
                
    demo.load(run_loop, inputs=None, outputs=[info_panel, traj_plot, latent_plot, log_panel])
    
    # Just set the flag, do not expect outputs from click
    sleep_btn.click(
        fn=controller.request_sleep,
        inputs=None,
        outputs=None
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, css=css, theme=gr.themes.Monochrome())
