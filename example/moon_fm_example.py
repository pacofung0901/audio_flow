import logging
import torch 
from torch import nn, Tensor
import time

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torchcfm.utils import *

start_time = time.time()

# Set up logging
logging.basicConfig(filename='training_log.txt', 
                    filemode='w',  # Append to the file
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.SELU(),
            nn.Linear(h, h), nn.SELU(),
            nn.Linear(h, h), nn.SELU(),
            nn.Linear(h, dim))
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step_midpoint(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        k1 = self(x_t=x_t, t=t_start)  # dx_t/dt at t_start
        midpoint_state = x_t + k1 * (dt / 2)  # x_t + dx_t/2
        t_mid = t_start + (dt / 2)
        k2 = self(x_t=midpoint_state, t=t_mid)  # d(x_t + dx_t/2)/d(t + dt/2)
        return x_t + k2 * dt

    def step_euler(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        # self(x_t=x_t, t=t_start) is dx_t/dt
        return x_t + self(x_t=x_t, t=t_start) * dt

    def step_heun(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        k1 = self(x_t=x_t, t=t_start) # dx_t/dt
        k2 = self(x_t=x_t + k1 * dt, t=t_start + dt) # d(x_t+dx_t)/d(x+dt)
        return x_t + (k1 + k2) * dt / 2 # average of them

flow = Flow()

optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for i in range(10000):
    x_1 = Tensor(make_moons(256, noise=0.05)[0])
    x_0 = torch.randn_like(x_1)
    t = torch.rand(len(x_1), 1)

    # find x_t and dx_t by x_0 and x_1
    x_t = (1 - t) * x_0 + t * x_1 # x_t = t * (x_1 - x_0) + x_0
    dx_t = x_1 - x_0 # dx_t/dt
    
    optimizer.zero_grad()
    # Calculate loss, so flow model is given t and x_t find dx_t/dt
    loss = loss_fn(flow(t=t, x_t=x_t), dx_t)

    # Log the loss
    logging.info(f'Epoch {i}, Loss: {loss.item()}')
    loss.backward()
    # print()
    optimizer.step()

x = torch.randn(300, 2)
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
time_steps = torch.linspace(0, 1.0, n_steps + 1)

axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f't = {time_steps[0]:.2f}')
axes[0].set_xlim(-3.0, 3.0)
axes[0].set_ylim(-3.0, 3.0)

for i in range(n_steps):
    x = flow.step_midpoint(x_t=x, t_start=time_steps[i], t_end=time_steps[i + 1])
    axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

plt.tight_layout()
# Save the figure as a PNG file
plt.savefig('moon_fm.png', dpi=300)  # Specify the file name and dpi
plt.show()

# Record the end time
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.4f} seconds")