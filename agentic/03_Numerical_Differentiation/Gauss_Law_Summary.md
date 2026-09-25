# Gauss's Law, Numerically

Goal: verify $\oint_S \mathbf E\cdot d\mathbf A = \int_V \nabla\cdot\mathbf E\,dV = Q_{\rm enc}$
with independent numerical routines. Units: $\varepsilon_0 = 1$.

## Files

| File | What it does | Run |
|---|---|---|
| `Gausslawproblem.py` | **LHS.** Point-charge and Gaussian-blob fields; midpoint quadrature of the flux over a sphere. | `python3 Gausslawproblem.py` |
| `Gausslawproblem_rhs.py` | **RHS.** $\nabla\cdot\mathbf E$ by central differences, integrated over the ball; compared with $\int\rho\,dV$ and the flux. | `python3 Gausslawproblem_rhs.py` |
| `lumpy_surface.py` | Same chain on a non-spherical surface $r = R[1 + a f(\hat n)]$; $a=0$ is the sphere. | `python3 lumpy_surface.py` |
| `gauss_viewer.py` | GUI: surface coloured by $\mathbf E\cdot\hat n$, E arrows, sliders for $R$ and bump amplitude $a$, live LHS / RHS / $Q_{\rm enc}$. | `python3 gauss_viewer.py` |

## Key results

- **Flux (LHS):** equals $Q_{\rm enc}$ for every configuration tried; the error falls as $h^2$ (fitted slope $-2$).
- **Charges outside:** zero net flux comes from cancellation (gross $\oint|\mathbf E\cdot\hat n|\,dA \neq 0$), not from zero field.
- **Divergence (RHS):** the step sweep shows a minimum near $h \approx \epsilon^{1/3} \approx 6\times10^{-6}$.
  Truncation error $\propto h^2$ (slope $+2$) dominates at large $h$; roundoff $\propto \epsilon/h$ (slope $-1$) dominates at small $h$.
- **Point charges break the RHS:** $\rho$ is a delta function the grid never samples, so $\int\nabla\cdot\mathbf E\,dV \approx 0$ while the flux is $q$.
- **Shape doesn't matter:** on the lumpy surface, only whether a charge is enclosed counts. A charge in a dent (inside the sphere, outside the surface) gives zero.
  For a set of mixed-sign blobs, flux, $\int\nabla\cdot\mathbf E\,dV$ and $\int\rho\,dV$ agree to $\sim 10^{-6}$.

## Open issue

`Gausslawproblem_rhs.py` check 5 fails. Its reference value $Q_{\rm enc}=1$ for the off-centre blob ignores the part of the Gaussian outside $R=2$. The numerics agree with each other ($\int\rho\,dV = 0.99963$), so the expected value needs fixing, not the code.
