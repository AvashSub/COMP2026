# Coupled Oscillators in a Box

Goal: simulate a 2-D network of masses and springs with different spring constants,
damping and applied forces, and find its resonances. This is the student's own project,
not the course brief in `Chaos_and_the_Butterfly_Effect.md`.
Units: $k = m = 1$, so frequencies are in units of $\sqrt{k/m}$.

## How this project is run

- The student writes a markdown cell describing the next step. Claude writes the code cell below it, short and readable.
- Don't add steps the markdown didn't ask for. Propose them instead.
- Every step gets a physical consistency check (an `assert` and a printed number), not just "it runs".
- The student edits the notebook in VS Code. Ask them to **save before Claude edits** the `.ipynb` and to **revert / reload from disk afterwards**. Saving the open copy overwrites Claude's changes, which happened twice.
- Commit notebooks with outputs cleared, because the two animations add about 20 MB.

## Trial 1 setup

Two masses in a square box, $W = H = 3$. Mass 0 rests at $(1, 1.5)$, mass 1 at $(2, 1.5)$.
There are 7 springs, and each mass is attached to 4 of them:

```
ends = [(0, 1),                                    # coupling spring between the masses
        ("left", 0), (0, "top"), (0, "bottom"),    # mass 0 to its walls
        (1, "right"), (1, "top"), (1, "bottom")]   # mass 1 to its walls
```

- Left and right springs attach at the mass's height. Top and bottom springs attach directly above and below it.
- Each rest length $L_0$ is the spring's length at rest, so no spring is stretched there. That gives $L_0 = [1, 1, 1.5, 1.5, 1, 1.5, 1.5]$.
- All $k = 1$ and both $m = 1$.

## Notebook `coupled_oscillator.ipynb`, cell by cell

| Cell | What it does |
|---|---|
| 1 | Classes. `box(W, H)` with `.anchor(wall, r)`. `spring_parameters(k, b, ends)` with `.attach(box, r0)`, which sets the wall anchors and $L_0$, plus `.endpoints(r)`, `.forces(r, v)` and `.energy(r)`. `masses(m)`. Spring ends are a mass index or a wall name; invalid input raises an error, with no silent defaults. |
| 2 | Trial 1 setup, above. Checks: 7 springs, 4 per mass, $L_0$ matches the geometry. |
| 4 | Static plot of the system at rest. `zigzag()` and `draw_system()` are used again by the animations. Check: zero net force at rest. |
| 6 | Markdown: the full nonlinear equations of motion and their small-oscillation form. |
| 7 | Undamped run, mass 0 driven in x with $F_0\cos\omega_d t$ ($F_0 = 0.2$, $\omega_d = 0.7$). Defines `animate(t, r_t, title)`, also used in cell 11. Check: work–energy theorem holds to ~$10^{-11}$. |
| 9 | Frequency sweep, mass 0 driven in x: $F_0 = 0.05$, $b = 0.05$ on every spring, $\omega_d$ from 0 to 3, 85 frequencies. Uses a damped copy `springs_d` and `linear_response(w)`, the small-oscillation steady state. **Fast mode:** each run starts from the linear steady state and settles for a fixed `T_settle = 60`, which takes ~10 s. |
| 11 | Mass 0 driven with $F_0\cos\omega_d t\,(\hat x + \hat y)$ at $\omega_d = \sqrt2$, with damping, then animated. Check: work–energy theorem with damping (drive work + damper work), and the dampers only ever remove energy. |

## Equations of motion

$$m_i\ddot{\mathbf r}_i = \sum_{s\ni i}\Big[-k_s(L_s-L_{0,s}) - b_s(\dot{\mathbf r}_i-\dot{\mathbf r}_j)\cdot\hat{\mathbf u}_s\Big]\hat{\mathbf u}_s + \mathbf F_i(t)$$

These are solved in full, nonlinear terms included, with `solve_ivp(method="DOP853")`. For small motions, x and y separate.
In x the coupling spring links the masses, with stiffness matrix $K = \begin{pmatrix}2&-1\\-1&2\end{pmatrix}$. In y each mass feels $2k$ and the masses are not linked.

## Key results

- **Natural frequencies:** x in-phase at $\omega = 1$, x out-of-phase at $\sqrt3$, and y at $\sqrt2$ for both masses.
- **Sweep, x drive on mass 0:** peaks at $\omega \approx 1.02$–$1.03$ and $1.74$, against 1 and $\sqrt3$. A dip near $\sqrt2$: mass 0 almost stops while mass 1 absorbs the drive.
  Off-peak the simulation matches linear theory to 0.5%. There is no y motion at all (exactly 0, by mirror symmetry).
- **Stiffening:** as a mass moves sideways, the springs perpendicular to its motion stretch and stiffen it. That bends the $\omega = 1$ peak to the right: it moves to ~1.03, drops ~8% below the linear height, and falls off sharply on the right.
  Near $\omega \approx 1.04$ the motion takes ~5× longer to settle than the damping predicts. So the fast-mode peak (1.020) is slightly off; a long, converged run gives 1.030.
- **Why $F_0 = 0.05$, not 1:** at $F_0 = 1$ the resonant amplitude is ~10, and the masses pass through the walls.
- **x + y drive at $\sqrt2$:** steady amplitudes are mass 0 $y \approx 0.32$ (linear 0.354) and mass 0 $x \approx 0.018$ (at the x dip). Mass 1 has $x \approx 0.067$ and $y \approx 0.060$.
  Mass 1's y motion is a nonlinear effect. Mass 0's motion stretches the coupling spring, so it now resists tilting and links the two y oscillators, which share the resonance $\sqrt2$.

## Possible next steps (not started)

- Different x and y drives: separate amplitudes, a phase offset (circular forcing), or separate frequencies.
- A y sweep, which should peak at $\sqrt2$, and a sweep with random $k_i$.
- More masses and a general lattice. The edge-list `ends` design already supports this. Forces are a Python loop over springs, so vectorize them if the network gets large.
