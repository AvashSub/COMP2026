"""Gauss's law on a half ball: a closed surface with a sharp edge.

    oint_S E . dA  =  int_V div E dV  =  Q_enc          (natural units, eps0 = 1)

S is the upper half ball |r| < R, z > 0: a dome (the upper hemisphere, outward
normal r-hat) plus a flat disk at z = 0 (outward normal -z-hat). The dome alone
is an open surface and obeys no such law -- its flux depends on where the
charges are, not just which are enclosed. Closing it with the disk restores the
law, and the two pieces meet at a circular edge where the normal jumps by 90
degrees, unlike the sphere and the lumpy surface.

Both pieces use midpoint sums: theta in [0, pi/2] and phi on the dome, r and
phi on the disk. Each piece is smooth up to its own boundary, so the edge is
never sampled and each sum is O(h^2) by itself.
"""

import numpy as np

from Gausslawproblem import point_charge, gaussian_blob, enclosed_charge, superpose
from Gausslawproblem_rhs import gaussian_density, divergence


# ---------------------------------------------------------------- the LHS

def dome_grid(R, n_theta, n_phi):
    """Midpoint nodes on the upper hemisphere: points, outward normals, areas."""
    dth, dph = (np.pi / 2) / n_theta, 2 * np.pi / n_phi
    TH, PH = np.meshgrid((np.arange(n_theta) + 0.5) * dth,
                         (np.arange(n_phi) + 0.5) * dph, indexing='ij')
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    return R * n, n, R**2 * np.sin(TH) * dth * dph


def disk_grid(R, n_r, n_phi):
    """Midpoint nodes on the disk z = 0, |r| < R, with outward normal -z-hat."""
    dr, dph = R / n_r, 2 * np.pi / n_phi
    RR, PH = np.meshgrid((np.arange(n_r) + 0.5) * dr,
                         (np.arange(n_phi) + 0.5) * dph, indexing='ij')
    pts = np.stack([RR * np.cos(PH), RR * np.sin(PH), np.zeros_like(RR)], axis=-1)
    n = np.broadcast_to([0.0, 0.0, -1.0], pts.shape)
    return pts, n, RR * dr * dph


def dome_flux(E, R, n=200):
    pts, nn, dA = dome_grid(R, n, 4 * n)
    return np.sum(np.sum(E(pts) * nn, axis=-1) * dA)


def disk_flux(E, R, n=200):
    pts, nn, dA = disk_grid(R, n, 4 * n)
    return np.sum(np.sum(E(pts) * nn, axis=-1) * dA)


def half_ball_flux(E, R, n=200):
    """Net outward flux through the closed half-ball surface. This is the LHS.

    n cells span the quarter circle in theta and the radius of the disk, and 4n
    cells go round in phi, so a cell is roughly square on both pieces.
    """
    return dome_flux(E, R, n) + disk_flux(E, R, n)


# ---------------------------------------------------------------- the volume

def half_ball_grid(R, n_r, n_mu, n_phi):
    """Nodes and weights for the half ball, dV = r^2 dr dmu dphi with mu in [0, 1].

    ball_grid() with the Gauss-Legendre mu nodes mapped to [0, 1] instead of
    [-1, 1]: the flat face is mu = 0, which Gauss-Legendre never samples.
    """
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    xm, wm = np.polynomial.legendre.leggauss(n_mu)
    rr, wrr = 0.5 * R * (xr + 1), 0.5 * R * wr           # [-1,1] -> [0,R]
    mu, wmu = 0.5 * (xm + 1), 0.5 * wm                   # [-1,1] -> [0,1]
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    wph = np.full(n_phi, 2 * np.pi / n_phi)

    RR, MU, PH = np.meshgrid(rr, mu, ph, indexing='ij')
    WR, WM, WP = np.meshgrid(wrr, wmu, wph, indexing='ij')
    st = np.sqrt(1 - MU**2)
    pts = np.stack([RR * st * np.cos(PH), RR * st * np.sin(PH), RR * MU], axis=-1)
    return pts, RR**2 * WR * WM * WP


def half_ball_integral(f, R, n_r=48, n_mu=48, n_phi=96):
    """Integrate a scalar field over the upper half ball."""
    pts, dV = half_ball_grid(R, n_r, n_mu, n_phi)
    return np.sum(f(pts) * dV)


def half_ball_divergence(E, R, **kw):
    """int_V (div E) dV -- the RHS, built from E alone."""
    return half_ball_integral(lambda r: divergence(E, r), R, **kw)


def inside(r0, R):
    r0 = np.asarray(r0, float)
    return np.linalg.norm(r0) < R and r0[2] > 0


# ---------------------------------------------------------------- checks

def check_closure():
    """Two fields whose flux through each piece is known exactly.

    E = z-hat: +pi R^2 out through the dome, -pi R^2 in through the disk, net 0.
    E = r:     div E = 3, so the net is 3V = 2 pi R^3; r is tangent to the disk,
               so all of it goes through the dome.
    """
    print("1. closure: constant field and E = r")
    R = 1.4
    Ez = lambda r: np.broadcast_to([0.0, 0.0, 1.0], r.shape)
    Er = lambda r: r
    dz, kz = dome_flux(Ez, R), disk_flux(Ez, R)
    dr_, kr = dome_flux(Er, R), disk_flux(Er, R)
    V = half_ball_integral(lambda r: np.ones(r.shape[:-1]), R)
    print("   E = z-hat:  dome=%+.9f  disk=%+.9f  (exact +-%.9f)  net=%+.1e"
          % (dz, kz, np.pi * R**2, dz + kz))
    print("   E = r:      dome=%+.9f  disk=%+.1e   3V=%.9f  (exact 2 pi R^3 = %.9f)"
          % (dr_, kr, 3 * V, 2 * np.pi * R**3))
    assert abs(kz + np.pi * R**2) < 1e-12, "disk area is wrong"
    assert abs(dz / (np.pi * R**2) - 1) < 1e-4 and abs(dz + kz) < 1e-4 * np.pi * R**2
    assert abs(kr) < 1e-15, "r should be tangent to the disk"
    assert abs(dr_ / (2 * np.pi * R**3) - 1) < 1e-4, "dome area is wrong"
    assert abs(V / (2 * np.pi * R**3 / 3) - 1) < 1e-12, "half-ball volume is wrong"
    print("   OK\n")


def check_disk_solid_angle():
    """The disk alone, against a closed form: flux of a point charge = q Omega / 4 pi.

    A charge on the axis at height z0 sees the disk under the solid angle
    Omega = 2 pi (1 - |z0| / sqrt(z0^2 + R^2)). The outward normal is -z-hat, so
    a charge above the disk sends flux out through it and one below sends it in.
    """
    print("2. disk alone vs the solid-angle formula")
    q, R = 1.0, 1.0
    for z0 in (0.3, 0.7, -0.3, -2.0):
        Omega = 2 * np.pi * (1 - abs(z0) / np.sqrt(z0**2 + R**2))
        expect = np.sign(z0) * q * Omega / (4 * np.pi)
        got = disk_flux(point_charge(q, [0, 0, z0]), R, n=400)
        print("   z0=%+.1f  disk flux=%+.7f  q Omega/4pi=%+.7f" % (z0, got, expect))
        assert abs(got - expect) < 1e-5, "disk flux disagrees with the solid angle"
    print("   OK\n")


def check_enclosure():
    """The dome alone is not Q_enc; dome + disk is.

    A charge in the lower half ball is inside the sphere but outside S: the dome
    still catches a sizeable flux, and the disk takes exactly that much back.
    """
    print("3. dome vs dome + disk")
    R = 1.0
    cases = [('upper half, on axis', 1.0, [0, 0, 0.3]),
             ('lower half, on axis', 1.0, [0, 0, -0.3]),
             ('upper half, off axis', -2.0, [0.5, -0.3, 0.4]),
             ('lower half, off axis', 1.5, [-0.4, 0.2, -0.6]),
             ('outside the sphere', 3.0, [0.0, 2.0, 1.0])]
    for label, q, r0 in cases:
        E = point_charge(q, r0)
        d, k = dome_flux(E, R, n=400), disk_flux(E, R, n=400)
        expect = q * inside(r0, R)
        print("   %-22s q=%+.1f  dome=%+.6f  disk=%+.6f  total=%+.7f  expect %+.1f"
              % (label, q, d, k, d + k, expect))
        assert abs(d + k - expect) < 1e-5, "total flux is not the enclosed charge"
    print("   the dome alone never equals q or 0 -- only the closed surface does")
    print("   OK\n")


def check_convergence_near_edge():
    """Error vs n for a charge far from the edge and one close to it.

    Each piece is a smooth midpoint sum, so both should reach slope -2. A charge
    near the edge makes E.n sharply peaked there, so the n it takes to reach
    the asymptotic slope is larger -- the edge costs resolution, not order.
    """
    print("4. convergence: charge far from and near the edge")
    ns = np.array([16, 32, 64, 128, 256, 512])
    slopes = {}
    for label, r0 in (('far (0.2, 0.1, 0.4)', [0.2, 0.1, 0.4]),
                      ('near (0.9, 0, 0.05)', [0.9, 0.0, 0.05])):
        E = point_charge(1.0, r0)
        errs = np.array([abs(half_ball_flux(E, 1.0, n) - 1.0) for n in ns])
        print("   %s: " % label + "  ".join("%.1e" % e for e in errs))
        slopes[label] = np.polyfit(np.log10(ns[-3:]), np.log10(errs[-3:]), 1)[0]
        print("      slope over the last three n: %.3f   (expect -2)" % slopes[label])
    assert all(abs(s + 2) < 0.2 for s in slopes.values()), "not second order"
    print("   OK\n")


def check_blobs():
    """The full chain with smooth charges.

    Centred blob: by symmetry exactly half its enclosed charge is in the upper
    half, a closed form. Blob straddling the disk: no closed form, so the three
    independent routines must agree with one another.
    """
    print("5. Gaussian blobs: flux, int div E, int rho")
    R = 1.0
    cases = [('centred, sigma=0.5', [(1.0, [0, 0, 0], 0.5)],
              0.5 * float(enclosed_charge(1.0, 0.5, R))),
             ('straddling the disk + far one', [(2.0, [0.3, -0.2, 0.1], 0.3),
                                                (-1.0, [0.0, 0.0, -2.5], 0.4)], None)]
    worst = 0.0
    for label, blobs, exact in cases:
        E = superpose(*[gaussian_blob(q, r0, s) for q, r0, s in blobs])
        rhos = [gaussian_density(q, r0, s) for q, r0, s in blobs]
        lhs = half_ball_flux(E, R, n=400)
        div = half_ball_divergence(E, R)
        chg = half_ball_integral(lambda r: sum(f(r) for f in rhos), R)
        print("   %-30s LHS=%+.8f  int div E=%+.8f  int rho=%+.8f  exact=%s"
              % (label, lhs, div, chg, '%+.8f' % exact if exact is not None else 'n/a'))
        worst = max(worst, abs(lhs - chg), abs(div - chg),
                    abs(chg - exact) if exact is not None else 0.0)
    print("   worst disagreement: %.1e" % worst)
    assert worst < 1e-5, "the chain disagrees"
    print("   OK\n")


if __name__ == '__main__':
    print("Gauss's law on a half ball,  units eps0 = 1\n")
    check_closure()
    check_disk_solid_angle()
    check_enclosure()
    check_convergence_near_edge()
    check_blobs()
