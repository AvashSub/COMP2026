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


def energy(net, x, V):
    psi, dpsi_dx = psi_and_grad(net, x)
    norm = torch.trapz(psi**2, x)
    kinetic = 0.5 * torch.trapz(dpsi_dx**2, x) / norm
    potential = torch.trapz(V(x) * psi**2, x) / norm
    return kinetic + potential


def normalized_psi(net, x):
    psi, _ = psi_and_grad(net, x)
    psi = psi.detach()
    norm = torch.sqrt(torch.trapz(psi**2, x))
    return psi / norm


def train(V, x, E_lower_bound, n_steps=2000, lr=1e-2, snapshot_steps=()):
    net = PsiNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    energy_history = []
    snapshots = []  # (step, E, normalized psi on x)

    for step in range(n_steps):
        optimizer.zero_grad()
        E = energy(net, x, V)
        E.backward()
        optimizer.step()

        # CHECK: variational principle is one-sided, E must never drop below E_lower_bound
        assert E.item() >= E_lower_bound - 1e-6, f"E={E.item():.6f} fell below {E_lower_bound} at step {step}"

        energy_history.append(E.item())
        if step in snapshot_steps:
            snapshots.append((step, E.item(), normalized_psi(net, x)))
        if step % 200 == 0:
            print(f"step {step:4d}  E = {E.item():.6f}")

    return net, energy_history, snapshots


def diagonalize(x, V):
    # H on the grid: -1/2 d^2/dx^2 by second-order finite differences, V on the diagonal.
    # Dirichlet walls at the grid edges, harmless while psi is negligible there.
    dx = (x[1] - x[0]).item()
    n = len(x)
    main = torch.full((n,), 1.0 / dx**2, dtype=torch.float64) + V(x).double()
    off = torch.full((n - 1,), -0.5 / dx**2, dtype=torch.float64)
    H = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)
    evals, evecs = torch.linalg.eigh(H)
    psi0 = evecs[:, 0]
    psi0 = psi0 / torch.sqrt(torch.trapz(psi0**2, x.double()))
    return evals[0].item(), psi0.float()


def overlap(psi_a, psi_b, x):
    # both normalized, so |<a|b>| = 1 only if they are the same state up to a phase
    return torch.trapz(psi_a * psi_b, x).abs().item()


def plot_snapshots(x, snapshots, exact_psi, path, exact_label):
    x_np = x.detach().numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    for step, E_step, psi_snap in snapshots:
        if psi_snap[psi_snap.abs().argmax()] < 0:
            psi_snap = -psi_snap
        ax.plot(x_np, psi_snap.numpy(), label=f"step {step}, E={E_step:.3f}")
    if exact_psi is not None:
        ax.plot(x_np, exact_psi.numpy(), "k--", label=exact_label)
    ax.set_xlabel("x")
    ax.set_ylabel(r"normalized $\psi(x)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    x = torch.linspace(-5.0, 5.0, 200)

    # sanity check the autodiff machinery once, on an untrained network
    net0 = PsiNet()
    psi0, dpsi0 = psi_and_grad(net0, x)
    dx = x[1] - x[0]
    fd = (psi0.detach()[2:] - psi0.detach()[:-2]) / (2 * dx)
    diff = (dpsi0.detach()[1:-1] - fd).abs().max().item()
    print(f"max |autodiff - finite diff| = {diff:.3e}")

    # --- harmonic oscillator: V(x) = 1/2 x^2, E_0 = 1/2 ---
    print("\n--- harmonic oscillator ---")
    E0_HARMONIC = 0.5
    V_harmonic = lambda x: 0.5 * x**2
    snapshot_steps = [0, 20, 50, 100, 500, 1999]

    net, energy_history, snapshots = train(V_harmonic, x, E0_HARMONIC, snapshot_steps=snapshot_steps)
    print(f"final E = {energy(net, x, V_harmonic).item():.6f}  (E_0 = {E0_HARMONIC})")

    # exact ground state, hbar = m = omega = 1: psi_0(x) = pi^-1/4 exp(-x^2/2)
    psi_exact_harmonic = torch.pi**-0.25 * torch.exp(-x**2 / 2)
    print(f"overlap |<psi_NN|psi_exact>| = {overlap(normalized_psi(net, x), psi_exact_harmonic, x):.6f}")

    # CHECK the diagonalizer itself, where the answer is known analytically,
    # before trusting it for the anharmonic case
    E0_fd, psi0_fd = diagonalize(x, V_harmonic)
    print(f"finite-difference E_0 = {E0_fd:.6f}  (exact {E0_HARMONIC})")
    print(f"overlap |<psi_fd|psi_exact>|  = {overlap(psi0_fd, psi_exact_harmonic, x):.6f}")

    plot_snapshots(
        x, snapshots, psi_exact_harmonic,
        "agentic/02_Neural_Networks/psi_training_snapshots.png",
        r"exact $\psi_0$ (Gaussian)",
    )

    # --- anharmonic oscillator: V(x) = 1/2 x^2 + lambda x^4 ---
    print("\n--- anharmonic oscillator ---")
    LAMBDA = 0.1
    V_anharmonic = lambda x: 0.5 * x**2 + LAMBDA * x**4

    # first-order perturbation theory: E_0 ~ 1/2 + lambda <x^4>_0, <x^4>_0 = 3/4
    E0_PERTURBATIVE = 0.5 + LAMBDA * 0.75

    net_anh, energy_history_anh, snapshots_anh = train(
        V_anharmonic, x, E0_HARMONIC, snapshot_steps=snapshot_steps
    )
    E_final_anh = energy(net_anh, x, V_anharmonic).item()
    print(f"final E = {E_final_anh:.6f}  (1st-order perturbative estimate = {E0_PERTURBATIVE:.6f})")

    # no closed form here, so diagonalization supplies the reference state
    E0_fd_anh, psi0_fd_anh = diagonalize(x, V_anharmonic)
    psi_nn_anh = normalized_psi(net_anh, x)
    print(f"finite-difference E_0 = {E0_fd_anh:.6f}")
    print(f"variational gap E_NN - E_0 = {E_final_anh - E0_fd_anh:.6f}")
    print(f"overlap |<psi_NN|psi_0>| = {overlap(psi_nn_anh, psi0_fd_anh, x):.6f}")
    assert E_final_anh >= E0_fd_anh - 1e-6, "variational energy fell below the diagonalized ground state"

    plot_snapshots(
        x, snapshots_anh, psi0_fd_anh,
        "agentic/02_Neural_Networks/psi_training_snapshots_anharmonic.png",
        r"diagonalized $\psi_0$",
    )
