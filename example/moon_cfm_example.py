import logging
import torch 
from torch import nn, Tensor
import time

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torchdyn.datasets import generate_moons


from torchdyn.core import NeuralODE
from torchcfm.models.models import *
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *

start_time = time.time()

# Set up logging
logging.basicConfig(filename='training_log.txt', 
                    filemode='w',  # Append to the file
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

def create_split_time_steps(time_steps, num_parts):
    """
    Create a tensor of equal splits from the original time_steps,
    ensuring the first and last indices are included without duplicates.

    Parameters:
    - time_steps: A tensor containing the original time steps (shape [n, 1]).
    - num_parts: The number of parts to split into.

    Returns:
    - A tensor of shape [num_parts, 1] containing the split time steps.
    - A tensor of shape [num_parts] containing the respective indices.
    """
    total_elements = time_steps.shape[0]  # Total number of elements in time_steps

    # Generate equal split indices
    indices = [int(i * (total_elements - 1) / (num_parts - 1)) for i in range(num_parts)]
    
    # Extract the split time steps and reshape to [num_parts, 1]
    split_time_steps = time_steps[indices].unsqueeze(1)

    return split_time_steps, torch.tensor(indices)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
sigma = 0.0
# flow = UNetModel(dim=2, num_channels=32, num_res_blocks=1).to(device)
flow = MLP(dim=2, time_varying=True)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
FM = ConditionalFlowMatcher(sigma=sigma)
# from IPython import embed; embed(using=False); import os; os._exit(0)

optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for i in range(10000):
    # x1 = Tensor(make_moons(256, noise=0.05)[0]) # sklearn moon
    x1 = sample_moons(256) # torchdyn moon


    # # fm, the source limited as Gaussian (8)
    # x0 = torch.randn_like(x1)
    # cfm, relex the source as 8 Gaussian, require more step (100)
    x0 = sample_8gaussians(256)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1) # ut = dx_t/dt (actual)
    vt = flow(torch.cat([xt, t[:, None]], dim=-1)) # vt = dx_t/dt (approximate)
    
    optimizer.zero_grad()
    loss = loss_fn(vt, ut) # mse

    # Log the loss
    logging.info(f'Epoch {i}, Loss: {loss.item()}')
    loss.backward()
    optimizer.step()

# # fm
# x = torch.randn(300, 2)
# cfm
x = sample_8gaussians(300)

n_steps = 100
time_steps = torch.linspace(0, 1.0, n_steps)
node = NeuralODE(torch_wrapper(flow), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

with torch.no_grad(): # [100, 300, 2]
    traj = node.trajectory(
            x,
            time_steps,
        )

# #################################
# #            CFM
# #################################
# plot with step
# num_parts = 8
# split_time_steps, indices = create_split_time_steps(time_steps, num_parts+1)

# fig, axes = plt.subplots(1, num_parts + 1, figsize=(30, 4), sharex=True, sharey=True)
# axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
# axes[0].set_title(f't = {split_time_steps[0].item():.2f}')
# axes[0].set_xlim(-5.0, 5.0)
# axes[0].set_ylim(-5.0, 5.0)
# for i, indice in enumerate(indices[1:]):
#     traj_step = traj.cpu().numpy()[indice]
#     axes[i + 1].scatter(traj_step[:, 0], traj_step[:, 1], s=10)
#     axes[i + 1].set_title(f't = {split_time_steps[i + 1].item():.2f}')
# plt.tight_layout()
# # Save the figure as a PNG file
# plt.savefig('moon_cfm.png', dpi=300)  # Specify the file name and dpi

# plot the path
plot_trajectories(traj.cpu().numpy())
plt.savefig('moon_cfm.png', dpi=300)


# Record the end time
end_time = time.time()

# Calculate the total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.4f} seconds")
