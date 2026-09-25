"""Interactive viewer for the Gauss's-law examples: the surface and E on it.

    oint_S E . dA  =  Q_enc          (natural units, eps0 = 1)

The surface is coloured by E.n, the outward flux density that the LHS sums, so
the LHS is literally the area-weighted total of the colour on the surface.
Arrows show the direction of E at a coarse set of surface points (fixed length,
same colour scale); their size is not |E|, which spans decades near a charge.

The surface is the lumpy one from lumpy_surface.py, r = R [1 + a f(n)] n: a = 0
is the sphere of the check scripts, and the amplitude slider deforms any example
to show that only the enclosed charge matters, never the shape.

The readout recomputes, for the current example, R and a:
    LHS       oint E.dA                    surface quadrature
    gross     oint |E.n| dA                what the net flux cancelled out of
    div       int (div E) dV               central differences + volume quadrature
    Q_enc     sum of enclosed point q + int rho dV over the volume for blobs
For point charges div E is a delta function the grid never sees, so the div
column misses their charge -- that is check 6 of the RHS script, not a bug.

The surface is always centred on the origin, as in both check scripts.
Run:  python3 gauss_viewer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

from Gausslawproblem import point_charge, gaussian_blob, superpose
from Gausslawproblem_rhs import gaussian_density
from lumpy_surface import surface_grid, surface_flux, surface_gross_flux, radius, \
    inside, volume_integral, volume_divergence, FUNKY_BLOBS, FUNKY_R, FUNKY_A

# name: (charges, default R, default a). A charge is (kind, q, position, sigma);
# sigma is ignored for point charges. Parameters are the ones used in the checks.
EXAMPLES = {
    'centred point charge': ([('point', 2.5, [0, 0, 0], None)], 1.0, 0.0),
    'point charge outside': ([('point', 3.0, [4, 0, 0], None)], 1.0, 0.0),
    'off-centre point charge': ([('point', 1.3, [0.9, 0, 0], None)], 1.0, 0.0),
    '4 charges, 2 in 2 out': ([('point', 2.0, [0.2, 0.1, 0.0], None),
                               ('point', -0.5, [-0.3, 0.0, 0.4], None),
                               ('point', 7.0, [3.0, 0.0, 0.0], None),
                               ('point', -9.0, [0.0, -8.0, 2.0], None)], 1.0, 0.0),
    'centred Gaussian blob': ([('blob', 1.0, [0, 0, 0], 1.0)], 1.5, 0.0),
    'blob off-centre': ([('blob', 1.0, [0.5, 0, 0], 0.4)], 2.0, 0.0),
    'blob entirely outside': ([('blob', 2.0, [5, 0, 0], 0.4)], 1.0, 0.0),
    'two blobs, +2 in, -3 out': ([('blob', 2.0, [0.3, 0.2, 0.0], 0.4),
                                  ('blob', -3.0, [6.0, 0.0, 0.0], 0.4)], 2.0, 0.0),
    'funky blobs, lumpy surface': ([('blob', q, r0, s) for q, r0, s in FUNKY_BLOBS],
                                   FUNKY_R, FUNKY_A),
}

N_THETA = 200                 # flux quadrature, as in the check scripts
N_SURF = (30, 60)             # coloured surface patches
N_ARROW = (8, 16)             # arrow sample points


def build(charges):
    """Field E, and enclosed charge as a function of R, for a list of charges."""
    fields = [point_charge(q, r0) if kind == 'point' else gaussian_blob(q, r0, s)
              for kind, q, r0, s in charges]
    def q_enc(R, a):
        total = 0.0
        for kind, q, r0, s in charges:
            if kind == 'point':
                total += q * inside(np.asarray(r0, float), R, a)
            else:
                total += volume_integral(gaussian_density(q, r0, s), R, a)
        return total
    return superpose(*fields), q_enc


fig = plt.figure(figsize=(13, 7.5))
ax = fig.add_axes([0.0, 0.08, 0.62, 0.9], projection='3d')
cax = fig.add_axes([0.60, 0.25, 0.012, 0.5])
rax = fig.add_axes([0.68, 0.52, 0.30, 0.42])
sax = fig.add_axes([0.14, 0.06, 0.38, 0.025])
aax = fig.add_axes([0.14, 0.02, 0.38, 0.025])
zax = fig.add_axes([0.68, 0.44, 0.30, 0.06])
tax = fig.add_axes([0.68, 0.02, 0.30, 0.40]); tax.axis('off')

radio = RadioButtons(rax, list(EXAMPLES))
slider = Slider(sax, 'size R', 0.1, 6.0, valinit=1.0)
amp = Slider(aax, 'bump amplitude a', 0.0, 0.9, valinit=0.0)
zoom = CheckButtons(zax, ['zoom to surface (hides far charges)'])
cmap = cm.coolwarm
cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax)
cbar.set_label(r'$E\cdot\hat n$  (outward flux density, clipped)')
state = {'name': 'centred point charge'}


def draw():
    charges = EXAMPLES[state['name']][0]
    E, q_enc = build(charges)
    R, a = slider.val, amp.val

    # surface: vertex grid for the mesh, face midpoints (= surface_grid nodes) for colour
    th = np.linspace(0, np.pi, N_SURF[0] + 1)
    ph = np.linspace(0, 2 * np.pi, N_SURF[1] + 1)
    TH, PH = np.meshgrid(th, ph, indexing='ij')
    nv = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    X, Y, Z = np.moveaxis(radius(nv, R, a)[..., None] * nv, -1, 0)
    pts, n, _ = surface_grid(R, a, *N_SURF)
    En = np.sum(E(pts) * n, axis=-1)

    # symmetric colour scale, clipped at the 98th percentile: near a point charge
    # E.n spikes by orders of magnitude and would wash out the rest of the sphere
    vmax = np.percentile(np.abs(En), 98)
    norm = colors.Normalize(-vmax, vmax)
    face = cmap(norm(np.pad(En, ((0, 1), (0, 1)), mode='edge')))

    ax.clear()
    ax.plot_surface(X, Y, Z, facecolors=face, alpha=0.55, shade=False,
                    rstride=1, cstride=1, linewidth=0.3, edgecolor=(0, 0, 0, 0.12))

    # arrows: direction of E, coloured by E.n on the same scale
    apts, an, _ = surface_grid(R, a, *N_ARROW)
    apts, an = apts.reshape(-1, 3), an.reshape(-1, 3)
    Ea = E(apts)
    u = Ea / np.linalg.norm(Ea, axis=-1, keepdims=True)
    c = cmap(norm(np.sum(Ea * an, axis=-1)))
    # a 3D quiver is one shaft + two head segments per arrow, coloured in that order
    ax.quiver(*apts.T, *u.T, length=0.3 * R, normalize=False,
              colors=np.concatenate([c, np.repeat(c, 2, axis=0)]), linewidth=1.2)

    far = max(np.linalg.norm(r0) + (3 * s if kind == 'blob' else 0)
              for kind, q, r0, s in charges)
    size = R * (1 + a)                     # |f| < 1, so the surface lies inside this
    lim = 1.5 * size if zoom.get_status()[0] else 1.15 * max(size, far)
    for kind, q, r0, s in charges:
        if np.max(np.abs(r0)) > lim:        # zoomed in: this charge is out of view
            continue
        col = 'red' if q > 0 else 'blue'
        ax.scatter(*r0, s=60 if kind == 'point' else 60 + 600 * s, c=col,
                   alpha=1.0 if kind == 'point' else 0.35, depthshade=False)
        ax.text(*np.add(r0, 0.15), '%+g' % q, color=col)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('%s,  R = %.2f,  a = %.2f' % (state['name'], R, a))
    cbar.mappable.set_norm(norm)

    lhs = surface_flux(E, R, a, n_theta=N_THETA, n_phi=2 * N_THETA)
    gross = surface_gross_flux(E, R, a, n_theta=N_THETA, n_phi=2 * N_THETA)
    div = volume_divergence(E, R, a)
    qe = q_enc(R, a)
    lines = ['LHS    oint E.dA      = %+.6f' % lhs,
             'gross  oint |E.n| dA  = %.6f' % gross,
             'RHS    int div E dV   = %+.6f' % div,
             'Q_enc                 = %+.6f' % qe,
             '',
             '|LHS - Q_enc| = %.2e' % abs(lhs - qe)]
    if any(kind == 'point' for kind, *_ in charges):
        lines += ['', 'point charges: div E is a delta', 'function, so int div E dV misses', 'their charge (RHS check 6)']
    # the flux quadrature needs the charge well off the surface; within a couple
    # of theta-cells of it the midpoint rule is no longer converged
    cell = np.pi * size / N_THETA
    def gap(r0):                            # radial distance from r0 to the surface
        d = np.linalg.norm(r0)
        return abs(d - radius(np.asarray(r0, float) / d, R, a)) if d > 0 else radius(np.zeros(3), R, a)
    if any(kind == 'point' and gap(r0) < 2 * cell for kind, q, r0, s in charges):
        lines += ['', 'WARNING: a point charge is on the', 'surface -- the flux is undefined']
    tax.clear(); tax.axis('off')
    tax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=10)
    fig.canvas.draw_idle()


def on_example(name):
    state['name'] = name
    _, R, a = EXAMPLES[name]
    amp.eventson = False                  # set both sliders, then draw once
    amp.set_val(a)
    amp.eventson = True
    slider.set_val(R)                     # triggers draw() via the slider callback


radio.on_clicked(on_example)
slider.on_changed(lambda _: draw())
amp.on_changed(lambda _: draw())
zoom.on_clicked(lambda _: draw())
draw()
plt.show()
