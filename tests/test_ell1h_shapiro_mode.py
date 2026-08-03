"""ELL1H H3+STIGMA Shapiro convention: full (Eq. 29) vs absorbed (Eq. 28)."""

from io import StringIO

import numpy as np
import pytest

from pint.models import get_model, get_model_and_toas
from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel
from pint.simulation import make_fake_toas_uniform


ELL1H_H3STIG = """
PSR J1234+5678
ELAT 0 1
ELONG 0 1
F0 100
PEPOCH 57000
BINARY ELL1H
PB 1.0
A1 10.0
TASC 57000
EPS1 1.0e-6
EPS2 1.0e-6
H3 1.0e-8
STIG 0.9
"""


ELL1H_H3H4 = """
PSR J1234+5678
ELAT 0 1
ELONG 0 1
F0 100
PEPOCH 57000
BINARY ELL1H
PB 1.0
A1 10.0
TASC 57000
EPS1 1.0e-6
EPS2 1.0e-6
H3 1.0e-6
H4 5.0e-7
"""


ELL1H_H3ONLY = """
PSR J1234+5678
ELAT 0 1
ELONG 0 1
F0 100
PEPOCH 57000
BINARY ELL1H
PB 1.0
A1 10.0
TASC 57000
EPS1 1.0e-6
EPS2 1.0e-6
H3 1.0e-6
"""


def _ell1h_with_phase(phi):
    bi = ELL1Hmodel()
    bi.H3 = 1.0e-8
    bi.STIGMA = 0.9
    bi.NHARMS = 7
    bi.Phi = lambda: phi  # noqa: E731
    return bi


def test_default_is_full_eq29():
    m = get_model(StringIO(ELL1H_H3STIG))
    bi = m.components["BinaryELL1H"].binary_instance
    assert bi.ds_func == bi.delayS_H3_STIGMA_exact
    assert m.meta["ell1h_shapiro"] == "full"


def test_absorbed_selects_eq28():
    m = get_model(StringIO(ELL1H_H3STIG), ell1h_shapiro="absorbed")
    bi = m.components["BinaryELL1H"].binary_instance
    assert bi.ds_func == bi.delayS3p_H3_STIGMA_exact
    assert m.meta["ell1h_shapiro"] == "absorbed"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="ell1h_shapiro"):
        get_model(StringIO(ELL1H_H3STIG), ell1h_shapiro="tempo2")


def test_setup_preserves_absorbed():
    m = get_model(StringIO(ELL1H_H3STIG), ell1h_shapiro="absorbed")
    m.setup()
    bi = m.components["BinaryELL1H"].binary_instance
    assert bi.ds_func == bi.delayS3p_H3_STIGMA_exact


def test_absorbed_minus_full_matches_gauge_term():
    bi = _ell1h_with_phase(np.linspace(0.0, 2.0 * np.pi, 256))
    h3 = bi.H3
    stig = bi.STIGMA
    m2 = h3 / stig**3
    phi = bi.Phi()
    d_full = bi.delayS_H3_STIGMA_exact(h3, stig, None)
    d_abs = bi.delayS3p_H3_STIGMA_exact(h3, stig, None)
    expected = -2.0 * m2 * (2.0 * stig * np.sin(phi) - stig**2 * np.cos(2.0 * phi))
    assert np.allclose(d_abs - d_full, expected, rtol=0.0, atol=1e-18)


def test_h3h4_unaffected_by_mode():
    m_full = get_model(StringIO(ELL1H_H3H4), ell1h_shapiro="full")
    m_abs = get_model(StringIO(ELL1H_H3H4), ell1h_shapiro="absorbed")
    bi_full = m_full.components["BinaryELL1H"].binary_instance
    bi_abs = m_abs.components["BinaryELL1H"].binary_instance
    assert bi_full.ds_func == bi_full.delayS3p_H3_STIGMA_approximate
    assert bi_abs.ds_func == bi_abs.delayS3p_H3_STIGMA_approximate


@pytest.mark.parametrize("mode", ["full", "absorbed"])
def test_h3_only_unaffected_by_mode(mode):
    m = get_model(StringIO(ELL1H_H3ONLY), ell1h_shapiro=mode)
    bi = m.components["BinaryELL1H"].binary_instance
    assert bi.ds_func == bi.delayS3p_H3_STIGMA_approximate
    assert m.meta["ell1h_shapiro"] == mode


def test_get_model_and_toas_forwards_absorbed(tmp_path):
    parpath = tmp_path / "psr.par"
    timpath = tmp_path / "psr.tim"
    parpath.write_text(ELL1H_H3STIG)
    m0 = get_model(StringIO(ELL1H_H3STIG))
    toas = make_fake_toas_uniform(57000, 57100, 8, m0, obs="ao")
    toas.write_TOA_file(timpath)
    m, _ = get_model_and_toas(parpath, timpath, ell1h_shapiro="absorbed")
    bi = m.components["BinaryELL1H"].binary_instance
    assert bi.ds_func == bi.delayS3p_H3_STIGMA_exact
    assert m.meta["ell1h_shapiro"] == "absorbed"


@pytest.mark.parametrize(
    "deriv_name, param_name, delta",
    [
        ("d_delayS3p_H3_STIGMA_exact_d_STIGMA", "STIGMA", 1e-8),
        ("d_delayS3p_H3_STIGMA_exact_d_H3", "H3", 1e-12),
        ("d_delayS3p_H3_STIGMA_exact_d_Phi", "Phi", 1e-8),
    ],
)
def test_absorbed_derivatives_match_finite_difference(deriv_name, param_name, delta):
    phi = np.linspace(0.1, 2.0 * np.pi - 0.1, 128)
    bi = _ell1h_with_phase(phi)
    h3 = float(bi.H3)
    stig = float(bi.STIGMA)

    analytic = getattr(bi, deriv_name)(h3, stig, None)

    if param_name == "STIGMA":
        d_plus = bi.delayS3p_H3_STIGMA_exact(h3, stig + delta, None)
        d_minus = bi.delayS3p_H3_STIGMA_exact(h3, stig - delta, None)
    elif param_name == "H3":
        d_plus = bi.delayS3p_H3_STIGMA_exact(h3 + delta, stig, None)
        d_minus = bi.delayS3p_H3_STIGMA_exact(h3 - delta, stig, None)
    else:  # Phi
        bi_plus = _ell1h_with_phase(phi + delta)
        bi_minus = _ell1h_with_phase(phi - delta)
        d_plus = bi_plus.delayS3p_H3_STIGMA_exact(h3, stig, None)
        d_minus = bi_minus.delayS3p_H3_STIGMA_exact(h3, stig, None)

    numerical = (d_plus - d_minus) / (2.0 * delta)
    # Relative tolerance on the bulk of the orbit; avoid exact zeros.
    mask = np.abs(analytic) > 1e-20 * np.max(np.abs(analytic))
    assert np.allclose(analytic[mask], numerical[mask], rtol=1e-5, atol=0.0)


def test_absorbed_d_delayS_d_par_stigma_matches_direct():
    """Integration through the design-matrix dispatch used for STIGMA."""
    m = get_model(StringIO(ELL1H_H3STIG), ell1h_shapiro="absorbed")
    toas = make_fake_toas_uniform(57000, 57100, 32, m, obs="ao")
    m.components["BinaryELL1H"].update_binary_object(toas)
    bi = m.components["BinaryELL1H"].binary_instance
    direct = bi.d_delayS3p_H3_STIGMA_exact_d_STIGMA(bi.H3, bi.STIGMA, None)
    via_par = bi.d_delayS_d_par("STIGMA")
    assert np.allclose(via_par, direct, rtol=0.0, atol=0.0)
