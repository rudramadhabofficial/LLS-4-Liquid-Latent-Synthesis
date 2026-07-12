import numpy as np
import math

class Environment:
    def __init__(self, size=20, max_energy=100.0):
        self.size = size
        self.max_energy = max_energy
        self.reset()
        
    def reset(self):
        # Agent state
        self.agent_pos = np.array([self.size / 2.0, self.size / 2.0])
        self.agent_vel = 0.0
        self.agent_angle = 0.0 # Radians
        self.energy = self.max_energy
        
        # Target state
        self.target_pos = self._random_pos()
        
        # Obstacle state (simple 1 obstacle for now)
        self.obstacle_pos = self._random_pos()
        
        self.done = False
        return self._get_state()
        
    def _random_pos(self):
        return np.random.rand(2) * self.size
        
    def _get_state(self):
        # [distance_to_target, distance_to_obstacle, agent_energy, current_velocity]
        dist_target = np.linalg.norm(self.agent_pos - self.target_pos)
        dist_obstacle = np.linalg.norm(self.agent_pos - self.obstacle_pos)
        
        return np.array([dist_target, dist_obstacle, self.energy, self.agent_vel], dtype=np.float32)
        
    def step(self, action):
        """
        action: [steering_angle, acceleration_force]
        steering_angle: [-pi, pi] approx
        acceleration_force: [-1, 1] approx
        """
        if self.done:
            return self._get_state(), 0.0, True, {}
            
        steering, force = action
        
        # Update angle and velocity
        self.agent_angle += float(steering)
        # clamp angle
        self.agent_angle = self.agent_angle % (2 * math.pi)
        
        self.agent_vel += float(force)
        self.agent_vel = np.clip(self.agent_vel, 0.0, 2.0) # max speed 2.0
        
        # Update position
        dx = math.cos(self.agent_angle) * self.agent_vel
        dy = math.sin(self.agent_angle) * self.agent_vel
        self.agent_pos += np.array([dx, dy])
        
        # Keep agent in bounds (bounce)
        if self.agent_pos[0] < 0 or self.agent_pos[0] > self.size:
            self.agent_pos[0] = np.clip(self.agent_pos[0], 0, self.size)
            self.agent_angle = math.pi - self.agent_angle
            self.agent_vel *= 0.5
            
        if self.agent_pos[1] < 0 or self.agent_pos[1] > self.size:
            self.agent_pos[1] = np.clip(self.agent_pos[1], 0, self.size)
            self.agent_angle = -self.agent_angle
            self.agent_vel *= 0.5
            
        # Drain energy
        energy_drain = 0.1 + (self.agent_vel * 0.5)
        self.energy -= energy_drain
        
        reward = 0.0
        
        # Check target
        dist_target = np.linalg.norm(self.agent_pos - self.target_pos)
        if dist_target < 1.0: # Reached target
            reward += 10.0
            self.target_pos = self._random_pos()
            self.energy = min(self.max_energy, self.energy + 20.0) # regain some energy
            
        # Check obstacle
        dist_obstacle = np.linalg.norm(self.agent_pos - self.obstacle_pos)
        if dist_obstacle < 1.0: # Hit obstacle
            reward -= 5.0
            self.agent_vel *= 0.2 # Dampen velocity
            
        # Time penalty
        reward -= 0.1
        
        if self.energy <= 0:
            self.done = True
            reward -= 10.0
            
        return self._get_state(), reward, self.done, {}

if __name__ == "__main__":
    env = Environment()
    s = env.reset()
    print("Initial state:", s)
    s, r, d, _ = env.step([0.1, 0.5])
    print("Next state:", s, "Reward:", r, "Done:", d)
