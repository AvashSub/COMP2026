import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Units: hbar = m = omega = 1

torch.manual_seed(0)


class PsiNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def psi_and_grad(net, x):
    x = x.clone().requires_grad_(True)
    psi = net(x.unsqueeze(-1))
    dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    return psi, dpsi_dx


def energy(net, x):
    psi, dpsi_dx = psi_and_grad(net, x)
    norm = torch.trapz(psi**2, x)
    kinetic = 0.5 * torch.trapz(dpsi_dx**2, x) / norm
    potential = 0.5 * torch.trapz(x**2 * psi**2, x) / norm
    return kinetic + potential


if __name__ == "__main__":
    net = PsiNet()
    x = torch.linspace(-5.0, 5.0, 200)

    psi, dpsi_dx = psi_and_grad(net, x)

    # CHECK: autodiff derivative vs. central finite difference on the same grid
    dx = x[1] - x[0]
    psi_np = psi.detach()
    fd = (psi_np[2:] - psi_np[:-2]) / (2 * dx)
    diff = (dpsi_dx.detach()[1:-1] - fd).abs().max().item()
    print(f"max |autodiff - finite diff| = {diff:.3e}")

    E_EXACT = 0.5
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for step in range(2000):
        optimizer.zero_grad()
        E = energy(net, x)
        E.backward()
        optimizer.step()

        # CHECK: variational principle is one-sided, E must never drop below E_0
        assert E.item() >= E_EXACT - 1e-6, f"E={E.item():.6f} fell below E_0={E_EXACT} at step {step}"

        if step % 200 == 0:
            print(f"step {step:4d}  E = {E.item():.6f}")

    print(f"final E = {energy(net, x).item():.6f}  (E_0 = {E_EXACT})")

    psi, dpsi_dx = psi_and_grad(net, x)
    x_np = x.detach().numpy()
    psi_np = psi.detach().numpy()
    dpsi_np = dpsi_dx.detach().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(x_np, psi_np)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\psi(x)$")
    ax2.plot(x_np, dpsi_np)
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$\psi'(x)$")
    fig.tight_layout()
    fig.savefig("agentic/02_Neural_Networks/variational_ground_state.png", dpi=150)
    print("saved plot to agentic/02_Neural_Networks/variational_ground_state.png")
