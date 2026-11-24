"""Read the odd-harmonic production fit used by the GSL pair diagnostics."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def read_production_template(template_path=None):
    """Return the frozen production coefficients from the milestone or an extracted copy."""
    if template_path is None:
        milestones = Path(__file__).resolve().parents[1] / "milestones"
        with ZipFile(milestones / "gsl_chirp_aware_phase_experiment_b1_w3.zip") as archive:
            template_path = BytesIO(archive.read("chirp_template_spp320.npz"))

    with np.load(template_path, allow_pickle=False) as data:
        coefficients = {key: data[key] for key in ("x", "CU", "CV", "CUx", "omega", "kappa")}
    n_points = len(coefficients["x"])
    if any(coefficients[key].shape != (9, n_points) for key in ("CU", "CV", "CUx")):
        raise ValueError("Expected a mean and cosine/sine pairs for harmonics 1, 3, 5, 7 (9 rows).")
    return coefficients
