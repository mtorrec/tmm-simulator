"""Pure TMM helpers used by the Streamlit UI in app.py.

No Streamlit imports here — this module is testable and reusable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d

import tmm


MaterialFn = Callable[[float], complex]


def list_nk_files(directory: str = "resources/nk data",
                  extensions: Sequence[str] = ('.txt', '.nk', '.dat')) -> List[str]:
    path = Path(directory)
    if not path.exists():
        return []
    files: List[str] = []
    for ext in extensions:
        files.extend(f.relative_to(path).as_posix() for f in path.rglob(f'*{ext}'))
    return sorted(files)


def load_nk_file(file) -> MaterialFn:
    """Return a function wl_nm -> n + ik from a path or Streamlit UploadedFile."""
    if isinstance(file, (str, os.PathLike)):
        data = None
        for encoding in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
            try:
                with open(file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                data_start = 0
                for i, line in enumerate(lines):
                    s = line.strip()
                    if s and not s.startswith('#') and s[0].isdigit():
                        data_start = i
                        break
                data = np.loadtxt(file, skiprows=data_start, encoding=encoding)
                break
            except (UnicodeDecodeError, ValueError, OSError):
                continue
        if data is None:
            raise ValueError(f"Could not load n-k file: {file}")
    else:
        try:
            data = np.loadtxt(file, skiprows=1)
        except Exception:
            file.seek(0)
            data = np.loadtxt(file)

    wavelengths = data[:, 0]
    n_data = data[:, 1]
    k_data = data[:, 2]

    # Auto unit conversion to nm
    if wavelengths.max() < 1e-6:
        wavelengths = wavelengths * 1e9
    elif wavelengths.max() < 1e-3:
        wavelengths = wavelengths * 1e3

    n_interp = interp1d(wavelengths, n_data, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    k_interp = interp1d(wavelengths, k_data, kind='linear',
                        bounds_error=False, fill_value='extrapolate')

    def _mat(wl):
        return n_interp(wl) + 1j * k_interp(wl)

    _mat.wl_min_data = float(wavelengths.min())
    _mat.wl_max_data = float(wavelengths.max())
    return _mat


def load_solar_spectrum(file) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wavelengths_nm, photon_flux [photons/cm^2/s/nm])."""
    data = np.loadtxt(file, comments='#')
    wavelengths = data[:, 0]
    irradiance = data[:, 1]  # W/m^2/nm

    h = 6.62607015e-34
    c = 2.99792458e8
    E_photon = h * c / (wavelengths * 1e-9)  # J
    flux_m2 = irradiance / E_photon  # photons/m^2/s/nm
    flux_cm2 = flux_m2 / 1e4
    return wavelengths, flux_cm2


def interpolate_spectrum(spec_wl: np.ndarray, spec_flux: np.ndarray,
                         target_wl: np.ndarray) -> np.ndarray:
    f = interp1d(spec_wl, spec_flux, kind='linear',
                 bounds_error=False, fill_value=0)
    return f(target_wl)


def _find_coh_stack_indices(layer_idx: int, c_list: Sequence[str]):
    if c_list[layer_idx] == 'i':
        return None, None
    stack_idx = 0
    layer_in_stack = 0
    in_current_stack = False
    for i in range(len(c_list)):
        if c_list[i] == 'c':
            if not in_current_stack:
                in_current_stack = True
                layer_in_stack = 0
            if i == layer_idx:
                return stack_idx, layer_in_stack + 1
            layer_in_stack += 1
        else:
            if in_current_stack:
                stack_idx += 1
                in_current_stack = False
    return None, None


def calculate_generation_tmm(materials: Sequence[MaterialFn],
                              thicknesses: Sequence[float],
                              coherency_list: Sequence[str],
                              active_layer_idx: int,
                              wavelengths: np.ndarray,
                              solar_flux: np.ndarray,
                              n_depths: int = 500,
                              angle_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute G(z,λ) [cm^-3 s^-1] in the active layer.

    Returns (z_positions_nm, generation_2d) where generation_2d has shape
    (n_depths, n_wavelengths).
    """
    active_thickness = thicknesses[active_layer_idx]
    z_positions = np.linspace(0, active_thickness, n_depths)

    coh_stack_idx, layer_in_coh_stack = _find_coh_stack_indices(
        active_layer_idx, coherency_list
    )
    use_inc = 'i' in coherency_list

    gen = np.zeros((len(wavelengths), n_depths))
    for i, (lam, flux) in enumerate(zip(wavelengths, solar_flux)):
        n_list = [m(lam) for m in materials]
        try:
            if use_inc:
                result = tmm.inc_tmm('s', n_list, thicknesses, coherency_list,
                                     angle_rad, lam)
            else:
                result = tmm.coh_tmm('s', n_list, thicknesses, angle_rad, lam)
        except Exception:
            continue

        row = np.zeros(n_depths)
        for j, z in enumerate(z_positions):
            try:
                if not use_inc:
                    data = tmm.position_resolved(active_layer_idx, z, result)
                    row[j] = data['absor']
                elif coh_stack_idx is not None and layer_in_coh_stack is not None:
                    stack_data = result['coh_tmm_data_list'][coh_stack_idx]
                    data = tmm.position_resolved(layer_in_coh_stack, z, stack_data)
                    row[j] = data['absor']
                # else: active layer is incoherent — leave zeros
            except Exception:
                row[j] = 0
        gen[i, :] = row * 1e7 * flux  # cm^-3 s^-1

    return z_positions, gen.T  # shape (n_depths, n_wavelengths)


def compute_rt_absorption(materials: Sequence[MaterialFn],
                           thicknesses: Sequence[float],
                           coherency_list: Sequence[str],
                           wavelengths: np.ndarray,
                           angle_rad: float = 0.0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightweight TMM: returns R(λ), T(λ), A(λ, layer).

    Does not resolve absorption with depth — suitable for sweeps where only
    integrated quantities are needed.
    """
    use_inc = 'i' in coherency_list
    nL = len(materials)
    R = np.zeros(len(wavelengths))
    T = np.zeros(len(wavelengths))
    A = np.zeros((len(wavelengths), nL))
    for i, lam in enumerate(wavelengths):
        n_list = [m(lam) for m in materials]
        try:
            if use_inc:
                res = tmm.inc_tmm('s', n_list, thicknesses, coherency_list,
                                  angle_rad, lam)
                layer_a = tmm.inc_absorp_in_each_layer(res)
            else:
                res = tmm.coh_tmm('s', n_list, thicknesses, angle_rad, lam)
                layer_a = tmm.absorp_in_each_layer(res)
            R[i] = float(res['R'])
            T[i] = float(res['T'])
            A[i, :] = np.array(layer_a, dtype=float)
        except Exception:
            # leave zeros in R, T, A row
            continue
    return R, T, A


def band_integrate(y_lambda: np.ndarray, flux: np.ndarray,
                   wavelengths: np.ndarray) -> float:
    """Photon-flux-weighted average of y(λ) over the given wavelength grid.

    Returns ∫ y·flux dλ / ∫ flux dλ. For y in [0,1] (R, T, A), result is also
    in [0,1] and interpretable as the fraction of incident photons.
    """
    num = np.trapezoid(y_lambda * flux, wavelengths)
    den = np.trapezoid(flux, wavelengths)
    if den <= 0:
        return 0.0
    return float(num / den)


def jgen_from_absorptance(absorptance_active: np.ndarray,
                           flux: np.ndarray,
                           wavelengths: np.ndarray) -> float:
    """Short-circuit current assuming SCE=1, in mA/cm².

    J = q · ∫ A(λ)·flux(λ) dλ, with flux in photons/cm²/s/nm.
    """
    q = 1.602176634e-19
    integral = np.trapezoid(absorptance_active * flux, wavelengths)  # /cm²/s
    return float(q * integral * 1e3)  # mA/cm²


def load_overlay_data(file) -> Tuple[np.ndarray, np.ndarray]:
    """Load the first 2 columns (wavelength_nm, value) from a text file for
    overlaying on plots.

    Robust to:
      - UTF-8 BOM;
      - '#'-prefixed comment blocks;
      - a column-header line that is NOT comment-prefixed (e.g. instrument
        exports: ``Wavelength [nm]\\tTransmittance\\tAbsorbance\\t...``) —
        scans for the first line whose first token parses as a float;
      - whitespace or comma delimiters;
      - extra columns beyond the first two (ignored);
      - NaN / inf rows (dropped);
      - percent vs. fraction scale (if max value > 2, divides by 100).

    Returns (wavelengths_nm, values_fraction).
    """
    import io as _io

    # 1. Read bytes/text and decode
    if hasattr(file, 'seek'):
        file.seek(0)
    if hasattr(file, 'read'):
        raw = file.read()
    else:
        with open(file, 'rb') as f:
            raw = f.read()
    if isinstance(raw, bytes):
        text = None
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError("Could not decode overlay file.")
    else:
        text = raw
    if text.startswith('﻿'):
        text = text[1:]

    # 2. Find first line whose first token is numeric (skip comments + headers)
    lines = text.splitlines()
    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        first_tok = s.replace(',', ' ').split()[0]
        try:
            float(first_tok)
            data_start = i
            break
        except ValueError:
            continue
    if data_start is None:
        raise ValueError("No numeric data found in overlay file.")

    data_text = '\n'.join(lines[data_start:])

    # 3. Parse: try whitespace then comma
    data = None
    for delim in (None, ','):
        try:
            d = np.loadtxt(_io.StringIO(data_text), comments='#',
                           delimiter=delim)
            if d.ndim == 2 and d.shape[1] >= 2:
                data = d
                break
            if d.ndim == 1 and d.size >= 2:
                data = d.reshape(1, -1)
                break
        except Exception:
            continue
    if data is None:
        raise ValueError(
            "Could not parse overlay file; need 2 numeric columns "
            "(wavelength, value)."
        )

    wl = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    mask = np.isfinite(wl) & np.isfinite(y)
    wl, y = wl[mask], y[mask]
    if y.size and np.nanmax(y) > 2:
        y = y / 100.0
    return wl, y
