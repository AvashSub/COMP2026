"""Gauss's law, right-hand side: the differential form.

    div E = rho          (natural units, eps0 = 1)

The divergence theorem turns the surface integral of the LHS into a volume one,
which gives a chain of four quantities that must all agree:

    oint_S E.dA   =   int_V (div E) dV   =   int_V rho dV   =   Q_enc
     surface          central differences     charge          exact erf
     quadrature       then volume quadrature  integral        formula

Three of those are independent numerical routines sharing no machinery, and the
fourth is closed form. The middle one is the real content of the differential
form: integrating rho by itself never touches E at all.

A point charge cannot be tested this way. Its rho is a delta function, so div E
vanishes everywhere a grid can see and the entire charge hides at one singular
point. Check 6 makes that failure explicit and measures it. The smooth Gaussian
blob is what the differential form can actually be checked against.
"""

import numpy as np
import matplotlib.pyplot as plt

from Gausslawproblem import (point_charge, gaussian_blob, enclosed_charge,
                             superpose, sphere_flux)

EPS = np.finfo(float).eps
H_OPT = EPS ** (1 / 3)        # ~6.1e-6: the central-difference optimum from notebook 3

# The blob used by the step-size sweep and by the figure, so the plot labels can
# quote exactly the rho that was computed.
BLOB_Q, BLOB_SIGMA, BLOB_R0 = 1.0, 1.0, (0.0, 0.0, 0.0)
SWEEP_PT = (0.5, -0.3, 0.7)   # where div E and rho are compared in the sweep


# ---------------------------------------------------------------- the source

def gaussian_density(q, r0, sigma):
    """rho of the same blob whose field gaussian_blob() returns.

    rho(r) = q (2 pi sigma^2)^{-3/2} exp(-|r - r0|^2 / 2 sigma^2), normalized so
    that its integral over all space is exactly q.
    """
    r0 = np.asarray(r0, float)
    def rho(r):
        d2 = np.sum((r - r0)**2, axis=-1)
        return q * (2 * np.pi * sigma**2)**-1.5 * np.exp(-d2 / (2 * sigma**2))
    return rho


# ---------------------------------------------------------------- differentiation

def divergence(E, r, h=H_OPT):
    """div E at points r by central differences: six field evaluations, O(h^2).

    Only E is used -- no analytic derivative anywhere -- so this is genuinely
    independent of the charge density it will be compared against.
    """
    r = np.asarray(r, float)
    out = np.zeros(r.shape[:-1])
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        out += (E(r + e)[..., i] - E(r - e)[..., i]) / (2 * h)
    return out


# ---------------------------------------------------------------- integration

def ball_grid(R, center, n_r, n_mu, n_phi):
    """Nodes and volume weights for a ball of radius R, with dV = r^2 dr dmu dphi.

    Gauss-Legendre in r and in mu = cos(theta): both integrands are smooth and
    non-periodic, where Gauss-Legendre converges far faster than a uniform rule.
    Uniform midpoint in phi, which is periodic and where that rule is already
    spectrally accurate.

    The point of spending accuracy here is to make the volume quadrature error
    negligible, so that whatever gap remains between the two sides of Gauss's
    law is the finite-difference error in div E and nothing else. Note also that
    Gauss-Legendre nodes are strictly interior, so r = 0 is never sampled.
    """
    center = np.asarray(center, float)
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    xm, wm = np.polynomial.legendre.leggauss(n_mu)
    rr, wrr = 0.5 * R * (xr + 1), 0.5 * R * wr          # map [-1,1] -> [0,R]
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    wph = np.full(n_phi, 2 * np.pi / n_phi)

    RR, MU, PH = np.meshgrid(rr, xm, ph, indexing='ij')
    WR, WM, WP = np.meshgrid(wrr, wm, wph, indexing='ij')
    st = np.sqrt(1 - MU**2)
    pts = center + np.stack([RR * st * np.cos(PH),
                             RR * st * np.sin(PH),
                             RR * MU], axis=-1)
    return pts, RR**2 * WR * WM * WP


def ball_integral(f, R, center=(0., 0., 0.), n_r=24, n_mu=24, n_phi=48):
    """Integrate a scalar field over the ball of radius R."""
    pts, dV = ball_grid(R, center, n_r, n_mu, n_phi)
    return np.sum(f(pts) * dV)


def volume_divergence(E, R, center=(0., 0., 0.), h=H_OPT, **kw):
    """int_V (div E) dV -- the RHS, built from E alone."""
    return ball_integral(lambda r: divergence(E, r, h), R, center, **kw)


# ---------------------------------------------------------------- checks

def check_quadrature_itself():
    """Establish the volume integrator before trusting it on physics.

    Two targets known exactly: the volume of a ball, and the total charge of a
    blob integrated out to where the tail is negligible. If either is off, no
    later disagreement could be attributed to the physics.
    """
    print("1. the volume integrator, on its own")
    for R in (0.7, 1.0, 3.0):
        vol = ball_integral(lambda r: np.ones(r.shape[:-1]), R)
        exact = 4 * np.pi * R**3 / 3
        print("   ball R=%.1f  volume=%.12f  exact=%.12f  rel.err=%.2e"
              % (R, vol, exact, abs(vol / exact - 1)))
        assert abs(vol / exact - 1) < 1e-13, "the volume quadrature is wrong"

    q, sigma = 1.7, 0.6
    rho = gaussian_density(q, [0, 0, 0], sigma)
    tot = ball_integral(rho, 8 * sigma, n_r=60, n_mu=24, n_phi=48)
    print("   blob charge out to R=8 sigma: %.12f   (q = %.4f, tail beyond is ~1e-14)"
          % (tot, q))
    assert abs(tot - q) < 1e-9, "the density does not integrate to its own charge"
    print("   OK\n")


def check_pointwise_divergence():
    """div E = rho, pointwise, at scattered points -- the differential law itself.

    E is an erf expression; rho is a bare Gaussian. The two are evaluated by
    completely separate code, and the derivative is taken numerically, so this
    is the differential form standing on its own with no integration involved.
    """
    print("2. div E vs rho, pointwise")
    q, sigma = 1.0, 1.0
    E, rho = gaussian_blob(q, [0, 0, 0], sigma), gaussian_density(q, [0, 0, 0], sigma)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.5, 2.5, size=(400, 3))
    d, a = divergence(E, pts), rho(pts)
    err = np.max(np.abs(d - a))
    print("   400 random points in a 5x5x5 box about the blob")
    print("   max |div E - rho| = %.3e     (peak rho = %.4f)" % (err, np.max(a)))
    print("   worst relative error where rho > 1%% of peak: %.2e"
          % np.max(np.abs((d - a)[a > 0.01 * np.max(a)] / a[a > 0.01 * np.max(a)])))
    assert err < 1e-9, "div E does not reproduce rho"
    print("   OK\n")


def check_step_size_sweep():
    """The U-curve: truncation falling as h^2, then roundoff climbing as 1/h.

    Notebook 3 measured this for a scalar derivative of sin. Here it is the same
    mechanism on a 3D vector field: div E needs six evaluations of nearby, nearly
    equal numbers, so catastrophic cancellation sets a floor no smaller step can
    beat. The fitted slope on the large-h side should be 2.
    """
    print("3. step-size sweep for div E -- truncation vs roundoff")
    E = gaussian_blob(BLOB_Q, BLOB_R0, BLOB_SIGMA)
    rho = gaussian_density(BLOB_Q, BLOB_R0, BLOB_SIGMA)
    pt = np.array([SWEEP_PT])
    exact = float(rho(pt)[0])

    hs = 10.0 ** np.arange(-1, -13, -1)
    errs = np.array([abs(float(divergence(E, pt, h)[0]) - exact) for h in hs])
    for h, e in zip(hs, errs):
        print("   h=%.0e   |div E - rho| = %.3e" % (h, e))

    i = int(np.argmin(errs))
    print("   best at h=%.0e (error %.2e); predicted optimum eps^(1/3)=%.0e"
          % (hs[i], errs[i], H_OPT))
    win = (hs >= 1e-3) & (hs <= 1e-1)
    slope = np.polyfit(np.log10(hs[win]), np.log10(errs[win]), 1)[0]
    print("   fitted slope over h in [1e-3, 1e-1]: %.3f   (expect 2)" % slope)

    assert abs(slope - 2) < 0.1, "the truncation error is not second order"
    assert errs[-1] > 100 * errs[i], "roundoff should blow up at tiny h"
    assert abs(np.log10(hs[i]) - np.log10(H_OPT)) <= 1.0, "optimum misplaced"
    print("   OK\n")
    return hs, errs, exact


def report(label, E, rho, R, q_exact, center=(0., 0., 0.), n_theta=400):
    """Print all four links of the chain for one configuration and return the gap."""
    lhs = sphere_flux(E, R, center=center, n_theta=n_theta, n_phi=2 * n_theta)
    div = volume_divergence(E, R, center=center)
    chg = ball_integral(rho, R, center=center)
    print("   %-34s  LHS=%+.9f  int div E=%+.9f  int rho=%+.9f  exact=%+.9f"
          % (label, lhs, div, chg, q_exact))
    return abs(lhs - div), abs(div - chg), abs(div - q_exact)


def check_chain_vs_radius():
    """All four quantities, as the sphere grows through the blob."""
    print("4. the full chain, blob of unit charge and width, R = 0.5 .. 5")
    q, sigma = 1.0, 1.0
    E, rho = gaussian_blob(q, [0, 0, 0], sigma), gaussian_density(q, [0, 0, 0], sigma)
    worst = 0.0
    for R in (0.5, 1.0, 2.0, 3.0, 5.0):
        g1, g2, g3 = report("R=%.1f" % R, E, rho, R, float(enclosed_charge(q, sigma, R)))
        worst = max(worst, g1, g2, g3)
    print("   worst disagreement anywhere in the chain: %.2e" % worst)
    assert worst < 1e-5, "the two sides of Gauss's law disagree"
    print("   OK\n")


def check_configurations():
    """Geometry that symmetry does not solve for you.

    Off-centre, wholly outside (both sides must vanish by cancellation), and a
    superposition where one blob is enclosed and one is not.
    """
    print("5. configurations where symmetry gives no shortcut")
    sigma = 0.4
    worst = 0.0

    E, rho = gaussian_blob(1.0, [0.5, 0, 0], sigma), gaussian_density(1.0, [0.5, 0, 0], sigma)
    g = report("blob off-centre, enclosed", E, rho, 2.0, 1.0)
    worst = max(worst, *g)

    E, rho = gaussian_blob(2.0, [5.0, 0, 0], sigma), gaussian_density(2.0, [5.0, 0, 0], sigma)
    g = report("blob entirely outside", E, rho, 1.0, 0.0)
    worst = max(worst, *g)

    E = superpose(gaussian_blob(2.0, [0.3, 0.2, 0.0], sigma),
                  gaussian_blob(-3.0, [6.0, 0.0, 0.0], sigma))
    rho = lambda r: (gaussian_density(2.0, [0.3, 0.2, 0.0], sigma)(r)
                     + gaussian_density(-3.0, [6.0, 0.0, 0.0], sigma)(r))
    g = report("two blobs, +2 in and -3 out", E, rho, 2.0, 2.0)
    worst = max(worst, *g)

    print("   worst disagreement: %.2e" % worst)
    assert worst < 1e-5, "the chain breaks when symmetry is removed"
    print("   OK\n")


def check_point_charge_pathology():
    """Where the differential form fails numerically, and exactly by how much.

    For a point charge div E = 0 identically away from the origin, and the origin
    is a set of measure zero that no grid ever lands on. So the RHS returns zero
    while the LHS returns q: the whole of Gauss's law is carried by a delta
    function the grid cannot represent. Then watch a blob shrink toward that
    limit and lose its charge as sigma drops below the node spacing.
    """
    print("6. the point-charge limit -- where the RHS stops working")
    q = 1.0
    E = point_charge(q, [0, 0, 0])
    lhs = sphere_flux(E, 1.0, n_theta=400, n_phi=800)
    rhs = volume_divergence(E, 1.0)
    print("   point charge:  LHS=%+.9f   int div E=%+.3e   missing=%.1f%% of q"
          % (lhs, rhs, 100 * (lhs - rhs) / q))
    assert abs(lhs - q) < 1e-5, "the LHS should be untroubled by the singularity"
    assert abs(rhs) < 1e-3, "div E away from a point charge is identically zero"

    print("   now a blob shrinking toward that limit (R=1, fixed 24-node radial grid):")
    for sigma in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        Eb = gaussian_blob(q, [0, 0, 0], sigma)
        rhs = volume_divergence(Eb, 1.0)
        frac = 100 * rhs / q
        print("     sigma=%.2f  dr/sigma=%.2f  int div E=%+.6f  (%6.2f%% of q recovered)"
              % (sigma, (1.0 / 24) / sigma, rhs, frac))
    print("   the law never fails; the grid does, once sigma falls below the spacing")
    print("   OK\n")


def make_figure(hs, errs, rho_exact):
    q, sigma, r0 = BLOB_Q, BLOB_SIGMA, BLOB_R0
    E = gaussian_blob(q, r0, sigma)
    Rs = np.linspace(0.3, 5.0, 18)
    lhs = np.array([sphere_flux(E, R, center=r0, n_theta=200, n_phi=400) for R in Rs])
    rhs = np.array([volume_divergence(E, R, center=r0) for R in Rs])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(r'source: Gaussian blob  $\rho(\mathbf{r}) = q\,(2\pi\sigma^2)^{-3/2}'
                 r'\,e^{-|\mathbf{r}-\mathbf{r}_0|^2/2\sigma^2}$,   '
                 r'$q=%g,\ \sigma=%g,\ \mathbf{r}_0=(%g,%g,%g)$,   '
                 r'$E$ = its exact erf field,   $\epsilon_0=1$' % ((q, sigma) + tuple(r0)))

    # panel 1: a single point, where rho is known exactly
    a1.loglog(hs, errs, 'o-', label='measured')
    # reference slopes, pinned to the data far from the crossover
    i2, i1 = np.argmin(abs(hs - 1e-2)), np.argmin(abs(hs - 1e-11))
    a1.loglog(hs, errs[i2] * (hs / hs[i2])**2, ':', c='gray', lw=1,
              label=r'$\propto h^2$ (truncation)')
    a1.loglog(hs, errs[i1] * (hs[i1] / hs), '--', c='gray', lw=1,
              label=r'$\propto \epsilon/h$ (roundoff)')
    a1.axvline(H_OPT, c='k', lw=0.8)
    a1.text(H_OPT / 1.5, errs.min() / 4, r'$h=\epsilon^{1/3}$', fontsize=9, ha='right')
    a1.set_ylim(errs.min() / 10, errs.max() * 1e4)
    a1.text(0.03, 0.97, r'at $\mathbf{r}=(%g,%g,%g)$' '\n'
            r'$|\mathbf{r}-\mathbf{r}_0|/\sigma=%.2f$' '\n'
            r'$\rho(\mathbf{r})=%.6f$ (exact)'
            % (SWEEP_PT + (np.linalg.norm(np.subtract(SWEEP_PT, r0)) / sigma, rho_exact)),
            transform=a1.transAxes, fontsize=8.5, va='top',
            bbox=dict(fc='white', ec='0.7', lw=0.6))
    a1.set_xlabel('step $h$')
    a1.set_ylabel(r'$|\nabla\cdot E(\mathbf{r}) - \rho(\mathbf{r})|$')
    a1.set_title(r'$\nabla\cdot E$ vs $\rho$ at one point')
    a1.legend(fontsize=8, loc='upper right')

    # panels 2-3: spheres of radius R centred on the blob
    note = r'spheres of radius $R$ centred on $\mathbf{r}_0$'
    a2.plot(Rs, lhs, 'o', ms=5, label=r'LHS  $\oint E\cdot dA$')
    a2.plot(Rs, rhs, 'x', ms=7, label=r'RHS  $\int \nabla\cdot E\,dV$')
    a2.plot(Rs, enclosed_charge(q, sigma, Rs), 'k-', lw=1.2,
            label=r'$Q_{\rm enc}=\int_{|\mathbf{r}-\mathbf{r}_0|<R}\rho\,dV$ (exact)')
    a2.text(0.97, 0.45, note, transform=a2.transAxes, ha='right', fontsize=8.5)
    a2.set_xlabel('$R/\\sigma$'); a2.set_ylabel('charge')
    a2.legend(fontsize=8.5, loc='lower right')
    a2.set_title('both sides vs the exact answer')

    a3.semilogy(Rs, np.abs(lhs - rhs), 'o-')
    a3.text(0.97, 0.05, note + '\n' + r'$\nabla\cdot E$ with $h=\epsilon^{1/3}$',
            transform=a3.transAxes, ha='right', fontsize=8.5)
    a3.set_xlabel('$R/\\sigma$')
    a3.set_ylabel(r'$|\oint E\cdot dA - \int\nabla\cdot E\,dV|$')
    a3.set_title('disagreement between the two sides')

    plt.tight_layout()
    fig.savefig('rhs_divergence_checks.png', dpi=140)
    print("wrote rhs_divergence_checks.png")


if __name__ == '__main__':
    print("Gauss's law RHS: div E = rho,  units eps0 = 1\n")
    check_quadrature_itself()
    check_pointwise_divergence()
    hs, errs, rho_exact = check_step_size_sweep()
    check_chain_vs_radius()
    check_configurations()
    check_point_charge_pathology()
    make_figure(hs, errs, rho_exact)
