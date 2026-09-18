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
    energy_history = []
    snapshot_steps = [0, 20, 50, 100, 500, 1999]
    snapshots = []  # (step, E, normalized psi on x)

    def normalized_psi(net, x):
        psi, _ = psi_and_grad(net, x)
        psi = psi.detach()
        norm = torch.sqrt(torch.trapz(psi**2, x))
        return psi / norm

    for step in range(2000):
        optimizer.zero_grad()
        E = energy(net, x)
        E.backward()
        optimizer.step()

        # CHECK: variational principle is one-sided, E must never drop below E_0
        assert E.item() >= E_EXACT - 1e-6, f"E={E.item():.6f} fell below E_0={E_EXACT} at step {step}"

        energy_history.append(E.item())
        if step in snapshot_steps:
            snapshots.append((step, E.item(), normalized_psi(net, x)))
        if step % 200 == 0:
            print(f"step {step:4d}  E = {E.item():.6f}")

    print(f"final E = {energy(net, x).item():.6f}  (E_0 = {E_EXACT})")

    # exact ground state, hbar = m = omega = 1: psi_0(x) = pi^-1/4 exp(-x^2/2)
    x_np_full = x.detach().numpy()
    psi_exact = torch.pi**-0.25 * torch.exp(-x**2 / 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for step, E_step, psi_snap in snapshots:
        if psi_snap[psi_snap.abs().argmax()] < 0:
            psi_snap = -psi_snap
        ax.plot(x_np_full, psi_snap.numpy(), label=f"step {step}, E={E_step:.3f}")
    ax.plot(x_np_full, psi_exact.numpy(), "k--", label=r"exact $\psi_0$ (Gaussian)")
    ax.set_xlabel("x")
    ax.set_ylabel(r"normalized $\psi(x)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("agentic/02_Neural_Networks/psi_training_snapshots.png", dpi=150)
    print("saved plot to agentic/02_Neural_Networks/psi_training_snapshots.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy_history)
    ax.axhline(E_EXACT, color="black", linestyle="--", label=r"$E_0$")
    ax.set_xlabel("training step")
    ax.set_ylabel("E")
    ax.legend()
    fig.tight_layout()
    fig.savefig("agentic/02_Neural_Networks/loss_curve.png", dpi=150)
    print("saved plot to agentic/02_Neural_Networks/loss_curve.png")

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
