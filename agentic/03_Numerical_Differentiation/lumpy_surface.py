"""Gauss's law on a surface that is not a sphere.

    oint_S E . dA  =  int_V div E dV  =  Q_enc          (natural units, eps0 = 1)

The surface is star-shaped about the origin: along each direction n(theta, phi)
it sits at

    r(theta, phi) = R [1 + a f(n)] n,

with f a fixed cubic polynomial in the components of n (three lobes round the
equator, a squash along z, and a twist). A polynomial in n is automatically
smooth on the sphere, poles included, and |f| <= 0.99, so any a < 1 keeps the
radius positive and the shape star-shaped. a = 0 is the sphere of radius R.

Surface element: dA = (dr/dtheta x dr/dphi) dtheta dphi, with the derivatives
taken analytically. For a = 0 this is R^2 sin(theta) n dtheta dphi, the sphere.
Volume: every ray from the origin leaves V exactly once, so V is the ball grid
with the radial coordinate stretched from [0, R] to [0, R(1 + a f)].
"""

import numpy as np

from Gausslawproblem import point_charge, gaussian_blob, superpose, sphere_grid
from Gausslawproblem_rhs import gaussian_density, divergence


# ---------------------------------------------------------------- the shape

def bumps(n):
    """f(n) and its gradient with respect to the components of n, shape (..., 3)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    f = 0.5 * (x**3 - 3 * x * y**2) + 0.15 * (3 * z**2 - 1) + x * y * z
    grad = np.stack([0.5 * (3 * x**2 - 3 * y**2) + y * z,
                     0.5 * (-6 * x * y) + x * z,
                     0.9 * z + x * y], axis=-1)
    return f, grad


def radius(n, R, a):
    """Distance from the origin to the surface along the unit vector n."""
    return R * (1 + a * bumps(n)[0])


def inside(r, R, a):
    """True where the point r lies strictly inside the surface."""
    s = np.linalg.norm(r, axis=-1)
    # at r = 0 the direction is undefined, but the origin is inside for any a < 1;
    # dividing by 1 there leaves n = 0 and radius > 0, which gives the same answer
    return s < radius(r / np.where(s > 0, s, 1)[..., None], R, a)


# ---------------------------------------------------------------- the LHS

def surface_grid(R, a, n_theta, n_phi):
    """Midpoint nodes on the lumpy surface: points, unit outward normals, areas.

    Same (theta, phi) midpoint grid and return convention as sphere_grid(), so the
    flux is sum(E.n dA) either way. With r = rho(theta, phi) n and rho = R(1 + a f),

        dr/dtheta = rho_theta n + rho n_theta,    dr/dphi = rho_phi n + rho n_phi,

    and rho_theta = R a grad f . n_theta (chain rule through the components of n).
    n_theta x n_phi = sin(theta) n points outward, so the cross product does too.
    """
    dth, dph = np.pi / n_theta, 2 * np.pi / n_phi
    TH, PH = np.meshgrid((np.arange(n_theta) + 0.5) * dth,
                         (np.arange(n_phi) + 0.5) * dph, indexing='ij')
    st, ct, sp, cp = np.sin(TH), np.cos(TH), np.sin(PH), np.cos(PH)
    n = np.stack([st * cp, st * sp, ct], axis=-1)
    n_th = np.stack([ct * cp, ct * sp, -st], axis=-1)
    n_ph = np.stack([-st * sp, st * cp, np.zeros_like(st)], axis=-1)

    f, grad = bumps(n)
    rho = R * (1 + a * f)
    rho_th = R * a * np.sum(grad * n_th, axis=-1)
    rho_ph = R * a * np.sum(grad * n_ph, axis=-1)
    r_th = rho_th[..., None] * n + rho[..., None] * n_th
    r_ph = rho_ph[..., None] * n + rho[..., None] * n_ph

    dA_vec = np.cross(r_th, r_ph) * dth * dph
    dA = np.linalg.norm(dA_vec, axis=-1)
    return rho[..., None] * n, dA_vec / dA[..., None], dA


def surface_flux(E, R, a, n_theta=200, n_phi=400):
    """Net outward flux of E through the lumpy surface. This is the LHS."""
    pts, n, dA = surface_grid(R, a, n_theta, n_phi)
    return np.sum(np.sum(E(pts) * n, axis=-1) * dA)


def surface_gross_flux(E, R, a, n_theta=200, n_phi=400):
    """Integral of |E.n| dA -- the size of the pieces the net flux cancels."""
    pts, n, dA = surface_grid(R, a, n_theta, n_phi)
    return np.sum(np.abs(np.sum(E(pts) * n, axis=-1)) * dA)


# ---------------------------------------------------------------- the volume

def volume_grid(R, a, n_r, n_mu, n_phi):
    """Nodes and weights for the region inside the surface.

    Write r = u rho(n) with u in [0, 1]; then dV = rho^3 u^2 du dmu dphi. Same
    Gauss-Legendre (u, mu) x midpoint (phi) product rule as ball_grid().
    """
    xu, wu = np.polynomial.legendre.leggauss(n_r)
    xm, wm = np.polynomial.legendre.leggauss(n_mu)
    uu, wuu = 0.5 * (xu + 1), 0.5 * wu                  # map [-1,1] -> [0,1]
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    wph = np.full(n_phi, 2 * np.pi / n_phi)

    U, MU, PH = np.meshgrid(uu, xm, ph, indexing='ij')
    WU, WM, WP = np.meshgrid(wuu, wm, wph, indexing='ij')
    st = np.sqrt(1 - MU**2)
    n = np.stack([st * np.cos(PH), st * np.sin(PH), MU], axis=-1)
    rho = radius(n, R, a)
    return (U * rho)[..., None] * n, rho**3 * U**2 * WU * WM * WP


def volume_integral(f, R, a, n_r=48, n_mu=48, n_phi=96):
    """Integrate a scalar field over the region inside the lumpy surface."""
    pts, dV = volume_grid(R, a, n_r, n_mu, n_phi)
    return np.sum(f(pts) * dV)


def volume_divergence(E, R, a, **kw):
    """int_V (div E) dV -- the RHS, built from E alone."""
    return volume_integral(lambda r: divergence(E, r), R, a, **kw)


# ---------------------------------------------------------------- the funky example

# Blobs of mixed sign and width, most straddling the a = 0.5 surface, one far
# outside. (q, centre, sigma). No closed form for Q_enc: the check is that three
# independent routines -- surface flux, volume div E, volume rho -- agree.
FUNKY_BLOBS = [(2.0, [0.9, 0.0, 0.0], 0.3),       # in the +x lobe, poking out
               (-0.8, [0.0, 0.0, 0.8], 0.25),     # at the flattened top
               (1.0, [-0.3, 0.4, -0.2], 0.5),     # wide, mostly inside
               (-1.0, [-0.5, -0.8, 0.0], 0.3),    # near a dent
               (4.0, [2.5, 1.0, 0.0], 0.3)]       # far outside
FUNKY_R, FUNKY_A = 1.0, 0.5


def funky_field():
    E = superpose(*[gaussian_blob(q, r0, s) for q, r0, s in FUNKY_BLOBS])
    rhos = [gaussian_density(q, r0, s) for q, r0, s in FUNKY_BLOBS]
    return E, lambda r: sum(rho(r) for rho in rhos)


# ---------------------------------------------------------------- checks

def check_sphere_limit():
    """a = 0 must reproduce sphere_grid() node for node, and the ball volume."""
    print("1. a = 0 is the sphere")
    p1, n1, d1 = surface_grid(1.3, 0.0, 40, 80)
    p2, n2, d2 = sphere_grid(1.3, (0, 0, 0), 40, 80)
    gap = max(np.max(np.abs(p1 - p2)), np.max(np.abs(n1 - n2)), np.max(np.abs(d1 / d2 - 1)))
    V = volume_integral(lambda r: np.ones(r.shape[:-1]), 1.3, 0.0)
    print("   max node/normal/area mismatch vs sphere_grid = %.1e" % gap)
    print("   volume = %.12f   4/3 pi R^3 = %.12f" % (V, 4 / 3 * np.pi * 1.3**3))
    assert gap < 1e-12, "a = 0 does not reduce to the sphere"
    assert abs(V / (4 / 3 * np.pi * 1.3**3) - 1) < 1e-12, "volume grid wrong at a = 0"
    print("   OK\n")


def check_closed_and_normals():
    """The surface is closed, and dA agrees with the volume grid.

    A constant field has zero net flux through any closed surface, so a gap or a
    flipped patch shows up as a nonzero sum. The field E = r has div E = 3, so its
    flux must be 3V, where V comes from the volume grid -- two routines that share
    only the function f, one using its gradient and the other not.
    """
    print("2. closed surface, and flux of E = r equals 3V")
    for a in (0.3, 0.6, 0.9):
        E0 = lambda r: np.broadcast_to([0.3, -1.1, 0.7], r.shape)
        net = surface_flux(E0, 1.0, a)
        gross = surface_gross_flux(E0, 1.0, a)
        V = volume_integral(lambda r: np.ones(r.shape[:-1]), 1.0, a)
        flux_r = surface_flux(lambda r: r, 1.0, a)
        print("   a=%.1f  constant E: net/gross=%.1e   flux(E=r)=%.9f  3V=%.9f  rel=%.1e"
              % (a, net / gross, flux_r, 3 * V, flux_r / (3 * V) - 1))
        assert abs(net / gross) < 1e-6, "surface is not closed"
        assert abs(flux_r / (3 * V) - 1) < 1e-4, "surface element disagrees with the volume"
    print("   OK\n")


def check_point_charges():
    """Only whether the charge is inside matters, not the shape.

    The 'dent' charge sits outside the lumpy surface but inside the sphere of
    the same R -- the case where a sphere-based guess would get it wrong.
    """
    print("3. point charges, lumpy surface a = 0.6")
    R, a = 1.0, 0.6
    cases = [('inside, off-centre', 1.7, [0.5, 0.2, -0.1]),
             ('outside, in a dent', 2.0, [-0.8, 0.0, 0.0]),
             ('inside a lobe, beyond R', -1.2, [1.15, 0.0, 0.0]),
             ('far outside', 5.0, [3.0, 3.0, 0.0])]
    for label, q, r0 in cases:
        r0 = np.array(r0, float)
        expect = q * inside(r0, R, a)
        flux = surface_flux(point_charge(q, r0), R, a, n_theta=400, n_phi=800)
        print("   %-26s |r0|=%.2f  surface there at %.2f   flux=%+.7f  expect %+.1f"
              % (label, np.linalg.norm(r0), radius(r0 / np.linalg.norm(r0), R, a), flux, expect))
        assert abs(flux - expect) < 1e-4, "flux does not see only the enclosed charge"
    print("   OK\n")


def check_convergence_order():
    """Midpoint in theta on a smooth surface: still O(h^2), slope -2."""
    print("4. convergence order on the lumpy surface")
    q, a = 1.0, 0.6
    E = point_charge(q, [0.4, 0.1, 0.0])
    ns = np.array([16, 32, 64, 128, 256])
    errs = np.array([abs(surface_flux(E, 1.0, a, n_theta=n, n_phi=2 * n) - q) for n in ns])
    for n, e in zip(ns, errs):
        print("   n_theta=%4d  |error|=%.3e" % (n, e))
    slope = np.polyfit(np.log10(ns), np.log10(errs), 1)[0]
    print("   fitted slope = %.3f   (expect -2)" % slope)
    assert abs(slope + 2) < 0.2, "not second order"
    print("   OK\n")


def check_funky_chain():
    """The full chain on the funky blobs: three independent numbers agree."""
    print("5. funky blobs on the lumpy surface, R=%.1f a=%.1f" % (FUNKY_R, FUNKY_A))
    E, rho = funky_field()
    lhs = surface_flux(E, FUNKY_R, FUNKY_A, n_theta=400, n_phi=800)
    div = volume_divergence(E, FUNKY_R, FUNKY_A)
    chg = volume_integral(rho, FUNKY_R, FUNKY_A)
    total = sum(q for q, _, _ in FUNKY_BLOBS)
    print("   LHS=%+.7f   int div E=%+.7f   int rho=%+.7f   (total charge %+.1f)"
          % (lhs, div, chg, total))
    print("   |LHS - int rho| = %.1e   |int div E - int rho| = %.1e"
          % (abs(lhs - chg), abs(div - chg)))
    assert abs(lhs - chg) < 1e-4 and abs(div - chg) < 1e-4, "the chain disagrees"
    print("   OK\n")


if __name__ == '__main__':
    print("Gauss's law on a lumpy surface,  units eps0 = 1\n")
    check_sphere_limit()
    check_closed_and_normals()
    check_point_charges()
    check_convergence_order()
    check_funky_chain()
