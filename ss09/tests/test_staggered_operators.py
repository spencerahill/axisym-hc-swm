"""Unit tests for the staggered-v (Arakawa C-grid) spatial operators.

v lives on the ny-1 interior cell faces y_i + dy/2; u and theta stay on the
ny centers. Three operators bridge the two grids in the momentum and
thermodynamic equations:

- v_faces_to_centers: average adjacent faces to centers (what u feels in the
  Coriolis and meridional-advection terms); wall centers carry v=0.
- v_divergence_at_centers: compact dv/dy at centers (the divergence in the
  vertical-advection terms), with the half-cell one-sided form at the walls.
- v_face_laplacian: the k_v diffusion of v on faces, a symmetric-association
  3-point Laplacian with mirror-image wall ghosts (v=0 at the wall).

Each operator is checked against hand-computed values AND for exact
floating-point parity on an antisymmetric input: reconstructed center-v and
the face Laplacian must be exactly antisymmetric, the divergence exactly
symmetric. Bit-exact parity is what lets the full staggered integration hold
the mirror-symmetry invariant the collocated model has.
"""

import numpy as np

from ss09.sw_model import (
    v_faces_to_centers,
    v_divergence_at_centers,
    v_face_laplacian,
)


def _antisymmetric_faces(nf, seed):
    """A face array f (length nf) with f[j] == -f[nf-1-j] exactly."""
    rng = np.random.default_rng(seed)
    half = rng.normal(0.0, 5.0, nf // 2)
    if nf % 2 == 0:
        return np.concatenate([half, -half[::-1]])
    # odd nf: the middle face sits at the equator and must be exactly 0
    return np.concatenate([half, [0.0], -half[::-1]])


# --- v_faces_to_centers ---------------------------------------------------

def test_v_faces_to_centers_hand_values():
    f = np.array([1.0, 2.0, 3.0, 4.0])  # nf=4 -> ny=5 centers
    vc = v_faces_to_centers(f)
    assert vc.shape == (5,)
    np.testing.assert_array_equal(vc, [0.0, 1.5, 2.5, 3.5, 0.0])


def test_v_faces_to_centers_walls_zero():
    f = np.array([7.0, -3.0, 5.0])
    vc = v_faces_to_centers(f)
    assert vc[0] == 0.0
    assert vc[-1] == 0.0


def test_v_faces_to_centers_antisymmetric_bitexact():
    for nf in (6, 7, 40, 41):
        f = _antisymmetric_faces(nf, seed=nf)
        vc = v_faces_to_centers(f)
        # v at centers is antisymmetric: vc == -vc[::-1], bit-exact
        assert np.max(np.abs(vc + vc[::-1])) == 0.0


# --- v_divergence_at_centers ----------------------------------------------

def test_v_divergence_at_centers_hand_values():
    f = np.array([1.0, 2.0, 3.0, 4.0])  # nf=4 -> ny=5, dy=2
    d = v_divergence_at_centers(f, dy=2.0)
    assert d.shape == (5,)
    # interior: (f[j]-f[j-1])/dy; walls: half-cell one-sided to v=0
    np.testing.assert_array_equal(d, [1.0, 0.5, 0.5, 0.5, -4.0])


def test_v_divergence_at_centers_antisymmetric_gives_symmetric_bitexact():
    for nf in (6, 7, 40, 41):
        f = _antisymmetric_faces(nf, seed=100 + nf)
        d = v_divergence_at_centers(f, dy=39377.5)
        # divergence of an antisymmetric field is symmetric: d == d[::-1]
        assert np.max(np.abs(d - d[::-1])) == 0.0


# --- v_face_laplacian -----------------------------------------------------

def test_v_face_laplacian_hand_values():
    f = np.array([1.0, 2.0, 3.0, 4.0])  # nf=4, dy=2 -> dy^2=4
    lap = v_face_laplacian(f, dy=2.0)
    assert lap.shape == (4,)
    # ghosts: fe = [-1, 1, 2, 3, 4, -4]; lap = ((fe[+1]+fe[-1]) - 2 fe[0]) / dy^2
    np.testing.assert_array_equal(lap, [-0.25, 0.0, 0.0, -2.25])


def test_v_face_laplacian_wall_ghost_is_mirror():
    """The outermost-face Laplacian uses a ghost = -f[0] (mirror about the
    wall to enforce v=0 there): lap[0] = (f[1] - 3 f[0]) / dy^2, NOT the
    phantom-zero form (f[1] - 2 f[0]) / dy^2."""
    f = np.array([2.0, 5.0, -1.0])
    lap = v_face_laplacian(f, dy=1.0)
    assert lap[0] == (f[1] - 3.0 * f[0])  # dy^2 = 1
    assert lap[-1] == (f[-2] - 3.0 * f[-1])


def test_v_face_laplacian_antisymmetric_bitexact():
    for nf in (6, 7, 40, 41):
        f = _antisymmetric_faces(nf, seed=200 + nf)
        lap = v_face_laplacian(f, dy=39377.5)
        # Laplacian of an antisymmetric field is antisymmetric, bit-exact
        assert np.max(np.abs(lap + lap[::-1])) == 0.0


def test_v_face_laplacian_symmetric_association_matters():
    """The naive association (f_plus - 2 f_center) + f_minus does NOT hold
    exact parity, which is why the symmetric (f_plus + f_minus) - 2 f_center
    form is used. On antisymmetric inputs the symmetric form is always
    bit-exact antisymmetric; the naive form drifts (~2e-18 per application,
    the seed of the patch's ~3.5e-10 parity error). Aggregated over seeds so
    the demonstration does not hinge on one lucky draw."""
    dy = 39377.5
    naive_drift_seen = 0.0
    for seed in range(30):
        f = _antisymmetric_faces(41, seed=seed)
        lap_sym = v_face_laplacian(f, dy=dy)
        fe = np.concatenate([[-f[0]], f, [-f[-1]]])
        lap_naive = ((fe[2:] - 2.0 * fe[1:-1]) + fe[:-2]) / dy**2
        # the symmetric form is exact for every draw
        assert np.max(np.abs(lap_sym + lap_sym[::-1])) == 0.0
        naive_drift_seen = max(
            naive_drift_seen, float(np.max(np.abs(lap_naive + lap_naive[::-1])))
        )
    # the naive association breaks parity somewhere across the draws
    assert naive_drift_seen > 0.0
