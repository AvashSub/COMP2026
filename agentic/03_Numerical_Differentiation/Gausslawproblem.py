"""Gauss's law, left-hand side: the integral form.

    oint_S E . dA  =  Q_enc

Convention: natural units with eps0 = 1, so the flux through a closed surface
equals the enclosed charge outright (no 1/eps0 anywhere).

Positions are numpy arrays of shape (..., 3). A field is a function mapping such
an array to another of the same shape, which makes superposition a one-liner.
The field of a point charge diverges at the charge, so no surface may pass
through one; that is a real restriction, not an edge case to paper over.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# ---------------------------------------------------------------- fields

def point_charge(q, r0):
    """E of a point charge q at r0."""
    r0 = np.asarray(r0, float)
    def E(r):
        d = r - r0
        s = np.linalg.norm(d, axis=-1, keepdims=True)
        return q * d / (4 * np.pi * s**3)
    return E


def gaussian_blob(q, r0, sigma):
    """E of total charge q smeared as a Gaussian of width sigma about r0.

    rho(r) = q (2 pi sigma^2)^{-3/2} exp(-r^2 / 2 sigma^2), whose charge inside
    radius s is enclosed_charge() below. Smooth everywhere, unlike a point charge.
    """
    r0 = np.asarray(r0, float)
    def E(r):
        d = r - r0
        s = np.linalg.norm(d, axis=-1, keepdims=True)
        return enclosed_charge(q, sigma, s) * d / (4 * np.pi * s**3)
    return E


def enclosed_charge(q, sigma, s):
    """Charge of a Gaussian blob inside radius s -- the exact answer for the blob."""
    u = s / sigma
    return q * (erf(u / np.sqrt(2)) - np.sqrt(2 / np.pi) * u * np.exp(-u**2 / 2))


def enclosed_charge_offset(q, sigma, r0, R):
    """Charge of a Gaussian blob centred at r0 inside the sphere |r| < R.

    With d = |r0| > 0 and a, b = (R - d)/sigma, (R + d)/sigma, the enclosed
    fraction is the CDF of a noncentral chi distribution with 3 degrees of freedom:

        (1/2)[erf(a/sqrt2) + erf(b/sqrt2)] - sigma/(d sqrt(2 pi)) [e^{-a^2/2} - e^{-b^2/2}].

    It tends to enclosed_charge() as d -> 0, but the second term is 0/0 at d = 0
    exactly, so a centred blob must use enclosed_charge() instead.
    """
    d = np.linalg.norm(r0)
    a, b = (R - d) / sigma, (R + d) / sigma
    return q * (0.5 * (erf(a / np.sqrt(2)) + erf(b / np.sqrt(2)))
                - sigma / (d * np.sqrt(2 * np.pi)) * (np.exp(-a**2 / 2) - np.exp(-b**2 / 2)))


def superpose(*fields):
    return lambda r: sum(f(r) for f in fields)


# ---------------------------------------------------------------- the LHS

# where in each step the integrand is sampled, as a fraction of the step
RULES = {'midpoint': 0.5,     # middle of each cell
         'euler': 0.0}        # left endpoint: Euler's method applied to an integral


def sphere_grid(R, center, n_theta, n_phi, rule='midpoint'):
    """Quadrature nodes on a sphere: points, outward normals, weights.

    Parametrize by (theta, phi); the outward normal is the radial unit vector and
    dA = R^2 sin(theta) dtheta dphi. Midpoint sampling keeps every node off the
    poles, so no node is special. The rule is O(h^2) in theta; in phi the
    integrand is periodic, where midpoint sampling converges far faster, so the
    theta grid alone sets the error.

    rule='euler' samples the left end of each step instead. Euler is normally
    O(h), but the sin(theta) in dA makes the integrand vanish at both theta = 0
    and theta = pi, and a left-endpoint sum with equal end values is exactly the
    trapezoid rule -- O(h^2), with error -2x the midpoint error (check 7).
    Its theta = 0 row sits on the pole, harmlessly: its weight is sin(0) = 0.
    """
    center = np.asarray(center, float)
    shift = RULES[rule]
    dth, dph = np.pi / n_theta, 2 * np.pi / n_phi
    TH, PH = np.meshgrid((np.arange(n_theta) + shift) * dth,
                         (np.arange(n_phi) + shift) * dph, indexing='ij')
    n = np.stack([np.sin(TH) * np.cos(PH),
                  np.sin(TH) * np.sin(PH),
                  np.cos(TH)], axis=-1)
    return center + R * n, n, np.sin(TH) * R**2 * dth * dph


def sphere_flux(E, R, center=(0., 0., 0.), n_theta=200, n_phi=400, rule='midpoint'):
    """Net outward flux of E through a sphere of radius R. This is the LHS."""
    pts, n, dA = sphere_grid(R, center, n_theta, n_phi, rule)
    return np.sum(np.sum(E(pts) * n, axis=-1) * dA)


def sphere_gross_flux(E, R, center=(0., 0., 0.), n_theta=200, n_phi=400):
    """Integral of |E.n| dA -- the size of the pieces the net flux cancels."""
    pts, n, dA = sphere_grid(R, center, n_theta, n_phi)
    return np.sum(np.abs(np.sum(E(pts) * n, axis=-1)) * dA)


# ---------------------------------------------------------------- checks

def check_centered_point_charge():
    """A centered point charge: flux = q, with a predictable discretization error.

    E.n is constant on the sphere, so the whole integral collapses to the midpoint
    rule for int_0^pi sin(theta) dtheta = 2. That rule's error is known in closed
    form, (h^2/24)[f'(pi) - f'(0)] = -h^2/12, so the flux must come out *high* by
    a relative pi^2 / (24 n_theta^2). Predicting the sign and size of the error is
    a much sharper test than just landing near q.
    """
    print("1. centered point charge -- flux and its predicted error")
    q = 2.5
    E = point_charge(q, [0, 0, 0])
    for n_theta in (25, 50, 100, 200):
        flux = sphere_flux(E, R=1.0, n_theta=n_theta, n_phi=2 * n_theta)
        rel = flux / q - 1
        pred = np.pi**2 / (24 * n_theta**2)
        print("   n_theta=%4d  flux=%.9f  rel.err=%+.3e  predicted=%+.3e  ratio=%.4f"
              % (n_theta, flux, rel, pred, rel / pred))
        assert rel > 0, "midpoint rule should overshoot here"
        assert abs(rel / pred - 1) < 0.01, "error does not match the closed-form prediction"
    print("   OK\n")


def check_radius_independence():
    """Flux is independent of R: E falls as 1/r^2 while area grows as r^2.

    Nothing enforces that cancellation numerically -- the two factors are computed
    by separate parts of the code -- so agreement over four decades in R is real
    evidence, not bookkeeping.
    """
    print("2. same charge, spheres from R=0.05 to R=500")
    q, Rs = -1.75, (0.05, 0.5, 5.0, 50.0, 500.0)
    E = point_charge(q, [0, 0, 0])
    fluxes = [sphere_flux(E, R=R, n_theta=100, n_phi=200) for R in Rs]
    for R, f in zip(Rs, fluxes):
        print("   R=%7.2f  flux=%.9f" % (R, f))
    spread = max(fluxes) - min(fluxes)
    print("   spread over all R = %.2e   (enclosed charge q = %.2f)" % (spread, q))
    assert spread < 1e-12, "flux should not depend on R"
    print("   OK\n")


def check_charge_outside():
    """A charge outside the surface contributes zero net flux.

    The integrand is large and of both signs -- field lines enter one side and
    leave the other -- so the answer is zero by cancellation, not because nothing
    was happening. Two things to establish: the net is tiny next to the gross
    |E.n| integral it cancelled out of, and what survives is only quadrature
    error, which must fall by 4x per doubling like everything else here.
    """
    print("3. charge outside the surface -- cancellation, not absence")
    E = point_charge(3.0, [4.0, 0, 0])
    gross = sphere_gross_flux(E, R=1.0, n_theta=200, n_phi=400)
    print("   gross integral of |E.n| dA = %.4e   (this is what must cancel)" % gross)

    prev = None
    for n in (50, 100, 200, 400):
        net = sphere_flux(E, R=1.0, n_theta=n, n_phi=2 * n)
        ratio = prev / net if prev else float('nan')
        print("   n_theta=%4d  net flux=%+.4e   net/gross=%.1e   drop vs prev=%.2f"
              % (n, net, abs(net / gross), ratio))
        assert abs(net / gross) < 1e-4, "net flux is not small next to what cancelled"
        if prev:
            assert 3.7 < ratio < 4.3, "residual is not falling at second order"
        prev = net
    print("   the residual is quadrature error, not physics: it -> 0 as O(h^2)")
    print("   OK\n")


def check_off_center_and_superposition():
    """Off-centre and multi-charge: only the enclosed charge counts.

    Moving the charge off-centre destroys the symmetry that made check 1 trivial --
    E.n now varies by orders of magnitude over the surface, and at d=0.95 the
    charge sits one twentieth of a radius from the surface it is being integrated
    over -- yet the answer never moves. Then several charges at once, two inside
    and two outside.
    """
    print("4. off-centre charge, then superposition")
    q = 1.3
    for d in (0.0, 0.3, 0.6, 0.9, 0.95):
        E = point_charge(q, [d, 0, 0])
        flux = sphere_flux(E, R=1.0, n_theta=400, n_phi=800)
        print("   charge at x=%.2f (R=1)  flux=%.9f  err=%+.2e" % (d, flux, flux - q))
        assert abs(flux - q) < 1e-5, "enclosed charge should not care where it sits"

    E = superpose(point_charge(2.0, [0.2, 0.1, 0.0]),     # inside
                  point_charge(-0.5, [-0.3, 0.0, 0.4]),   # inside
                  point_charge(7.0, [3.0, 0.0, 0.0]),     # outside
                  point_charge(-9.0, [0.0, -8.0, 2.0]))   # outside
    flux = sphere_flux(E, R=1.0, n_theta=400, n_phi=800)
    print("   4 charges (2 in, 2 out): flux=%.9f   expected Q_enc=%.1f" % (flux, 1.5))
    print("   the 16.0 of charge sitting outside contributes nothing")
    assert abs(flux - 1.5) < 1e-5, "flux should see only the enclosed charge"
    print("   OK\n")


def check_partial_enclosure():
    """A blob straddling the surface: flux tracks a nontrivial known function of R.

    For a Gaussian blob the enclosed charge is an erf expression rising from 0 to
    q. The flux integral has to reproduce that whole curve, so the target is a
    function, not a constant -- the sharpest test available on the LHS alone.
    """
    print("5. Gaussian blob straddling the surface -- flux vs exact Q_enc(R)")
    q, sigma = 1.0, 1.0
    E = gaussian_blob(q, [0, 0, 0], sigma)
    worst = 0.0
    for R in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        flux = sphere_flux(E, R=R, n_theta=300, n_phi=600)
        exact = float(enclosed_charge(q, sigma, R))
        worst = max(worst, abs(flux - exact))
        print("   R=%.2f  flux=%.9f  exact Q_enc=%.9f  err=%+.2e  (%5.1f%% of q enclosed)"
              % (R, flux, exact, flux - exact, 100 * exact / q))
    print("   worst error over the sweep = %.2e" % worst)
    assert worst < 1e-5, "flux must follow the analytic enclosed-charge curve"
    print("   OK\n")


def check_convergence_order():
    """The log-log slope of the error is the order of the quadrature.

    Same measurement as the finite-difference sweep in notebook 3, applied now to
    a 2D quadrature: fit the slope where the plot is straight. Midpoint in theta
    is O(h^2), so the slope against n_theta should be -2.
    """
    print("6. convergence order of the surface quadrature")
    q = 1.0
    E = point_charge(q, [0.4, 0.0, 0.0])      # off-centre: no symmetry shortcut
    ns = np.array([8, 16, 32, 64, 128, 256])
    errs = np.array([abs(sphere_flux(E, R=1.0, n_theta=n, n_phi=2 * n) - q) for n in ns])
    for n, e in zip(ns, errs):
        print("   n_theta=%4d  |error|=%.3e" % (n, e))
    slope = np.polyfit(np.log10(ns), np.log10(errs), 1)[0]
    print("   fitted slope = %.4f   (expect -2 for the midpoint rule)" % slope)
    assert abs(slope + 2) < 0.1, "this is not a second-order quadrature"
    print("   OK\n")
    return ns, errs, slope


def check_euler_vs_midpoint():
    """Euler's method (left endpoint) against the midpoint rule, same grid.

    Centred charge: E.n is constant, so the flux reduces to the rule applied to
    int_0^pi sin(theta) dtheta = 2. Left endpoint with f(0) = f(pi) = 0 is the
    trapezoid rule, whose error (-h^2/12)[f'(pi) - f'(0)] = +h^2/6 predicts a
    relative error of exactly -pi^2 / (12 n_theta^2): Euler undershoots, where
    midpoint overshoots by half as much (check 1). Off-centre, there is no closed
    form, but the ratio of the two errors must still tend to -2, and both must
    fit a slope of -2.
    """
    print("7. Euler (left endpoint) vs midpoint")
    q = 1.0
    E0, E1 = point_charge(q, [0, 0, 0]), point_charge(q, [0.4, 0.0, 0.0])
    ns = np.array([16, 32, 64, 128, 256])
    eu, mid = [], []
    for n in ns:
        rel = sphere_flux(E0, 1.0, n_theta=n, n_phi=2 * n, rule='euler') / q - 1
        pred = -np.pi**2 / (12 * n**2)
        eu.append(sphere_flux(E1, 1.0, n_theta=n, n_phi=2 * n, rule='euler') - q)
        mid.append(sphere_flux(E1, 1.0, n_theta=n, n_phi=2 * n) - q)
        print("   n_theta=%4d  centred: rel.err=%+.3e  predicted=%+.3e  |  off-centre: "
              "euler=%+.3e  midpoint=%+.3e  ratio=%+.4f"
              % (n, rel, pred, eu[-1], mid[-1], eu[-1] / mid[-1]))
        assert abs(rel / pred - 1) < 0.01, "Euler error does not match the trapezoid prediction"
    eu, mid = np.array(eu), np.array(mid)
    s_eu = np.polyfit(np.log10(ns), np.log10(np.abs(eu)), 1)[0]
    s_mid = np.polyfit(np.log10(ns), np.log10(np.abs(mid)), 1)[0]
    print("   fitted slopes: euler %.3f, midpoint %.3f   (expect -2 for both)" % (s_eu, s_mid))
    assert abs(s_eu + 2) < 0.1 and abs(s_mid + 2) < 0.1, "not second order"
    assert abs(eu[-1] / mid[-1] + 2) < 0.01, "error ratio should tend to -2"
    print("   same order; midpoint is 2x more accurate for the same work")
    print("   OK\n")


def make_figure(ns, errs, slope):
    q, sigma = 1.0, 1.0
    E = gaussian_blob(q, [0, 0, 0], sigma)
    Rs = np.linspace(0.05, 5.0, 60)
    flux = np.array([sphere_flux(E, R=R, n_theta=80, n_phi=160) for R in Rs])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(Rs, flux, 'o', ms=4, label='flux integral (LHS)')
    a1.plot(Rs, enclosed_charge(q, sigma, Rs), 'k-', lw=1.5, label=r'exact $Q_{\rm enc}(R)$')
    a1.axhline(q, ls=':', c='gray')
    a1.set_xlabel(r'sphere radius $R/\sigma$'); a1.set_ylabel('charge')
    a1.set_title(r'Gaussian blob: flux tracks $Q_{\rm enc}$'); a1.legend()

    a2.loglog(ns, errs, 'o-', label='measured')
    a2.loglog(ns, errs[0] * (ns / ns[0]) ** -2.0, 'k--', label=r'$n^{-2}$')
    a2.set_xlabel(r'$n_\theta$'); a2.set_ylabel('|flux error|')
    a2.set_title('quadrature order: slope %.2f' % slope); a2.legend()

    plt.tight_layout()
    fig.savefig('lhs_flux_checks.png', dpi=140)
    print("wrote lhs_flux_checks.png")


if __name__ == '__main__':
    print("Gauss's law LHS: oint E.dA = Q_enc,  units eps0 = 1\n")
    check_centered_point_charge()
    check_radius_independence()
    check_charge_outside()
    check_off_center_and_superposition()
    check_partial_enclosure()
    ns, errs, slope = check_convergence_order()
    check_euler_vs_midpoint()
    make_figure(ns, errs, slope)
