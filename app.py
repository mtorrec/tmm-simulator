"""TMM Simulator — Streamlit app.

Two tabs on top of a shared stack-definition UI:
  1. Single-point G(z,λ) analysis (the full plot grid from the original generator).
  2. Thickness sweep over a chosen layer, with multi-select output picker.

Backend lives in tmm_core.py (pure, no Streamlit imports).
"""

from __future__ import annotations

import io
import os
import time
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tmm
from plotly.subplots import make_subplots

from tmm_core import (
    band_integrate,
    calculate_generation_tmm,
    compute_rt_absorption,
    interpolate_spectrum,
    jgen_from_absorptance,
    list_nk_files,
    load_nk_file,
    load_solar_spectrum,
)


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="TMM Simulator", layout="wide")
st.title("TMM Simulator")
st.markdown(
    "Transfer-matrix generation profiles and thickness sweeps for thin-film "
    "solar stacks. Built on the [`tmm`](https://github.com/sbyrnes321/tmm) "
    "package by Steven Byrnes and compatible with "
    "[SIMsalabim](https://github.com/kostergroup/SIMsalabim) generation files."
)


# =============================================================================
# Session-state initialization — default perovskite stack
# =============================================================================

if 'layer_order' not in st.session_state:
    st.session_state.layer_order = [0, 1, 2, 3, 4, 5]
    st.session_state.layer_configs = {
        0: {'name': 'Glass',      'thickness': 1_000_000.0, 'file': None,
            'file_source': 'local', 'local_file_name': 'nk_glass.txt',
            'coherency': 'i'},
        1: {'name': 'FTO',        'thickness': 400.0,       'file': None,
            'file_source': 'local', 'local_file_name': 'nk_FTO.txt',
            'coherency': 'c'},
        2: {'name': 'SAM',        'thickness': 5.0,         'file': None,
            'file_source': 'local', 'local_file_name': 'dasgupta/other/SAM.nk',
            'coherency': 'c'},
        3: {'name': 'Perovskite', 'thickness': 500.0,       'file': None,
            'file_source': 'local',
            'local_file_name': 'dasgupta/FACsPb_BrI/FACsPb_Br010_Eg1.48eV.nk',
            'coherency': 'c'},
        4: {'name': 'C60',        'thickness': 30.0,        'file': None,
            'file_source': 'local', 'local_file_name': 'nk_C60_2.txt',
            'coherency': 'c'},
        5: {'name': 'Metal',      'thickness': np.inf,      'file': None,
            'file_source': 'local', 'local_file_name': 'nk_Au.txt',
            'coherency': 'c'},
    }
    st.session_state.active_layer_selector = 3  # Perovskite


# =============================================================================
# Shared: Stack-definition UI
# =============================================================================

st.markdown("### Stack definition")
st.caption(
    "Define layers from front (illumination side) to back. Air is prepended "
    "automatically. Mark thick layers (>1 µm, e.g. glass) as incoherent to "
    "remove interference fringes."
)

current_n_layers = len(st.session_state.layer_order) or 6
n_layers = st.number_input(
    "Number of layers (excluding air)",
    min_value=1, max_value=12, value=current_n_layers, step=1,
)

# Reconcile layer_order with n_layers
if len(st.session_state.layer_order) != n_layers:
    old_order = st.session_state.layer_order
    new_order = list(old_order[:n_layers])
    next_id = (max(old_order) if old_order else -1) + 1
    while len(new_order) < n_layers:
        new_order.append(next_id)
        next_id += 1
    removed = set(old_order) - set(new_order)
    for rid in removed:
        st.session_state.layer_configs.pop(rid, None)
    st.session_state.layer_order = new_order

for layer_id in st.session_state.layer_order:
    if layer_id not in st.session_state.layer_configs:
        st.session_state.layer_configs[layer_id] = {
            'name': f'Layer_{layer_id}',
            'thickness': 100.0,
            'file': None,
            'file_source': 'local',
            'local_file_name': '',
            'coherency': 'c',
        }

layers: List[dict] = [None] * len(st.session_state.layer_order)
nk_files = list_nk_files("resources/nk data")

st.markdown("**Layers** (⬆/⬇ to reorder, ✕ to delete)")
for display_idx, layer_id in enumerate(st.session_state.layer_order):
    cfg = st.session_state.layer_configs[layer_id]

    col_up, col_body, col_down, col_del = st.columns([0.6, 10, 0.6, 0.6])

    with col_up:
        if display_idx > 0:
            if st.button("⬆", key=f"up_{layer_id}", help="Move up"):
                order = st.session_state.layer_order
                order[display_idx - 1], order[display_idx] = \
                    order[display_idx], order[display_idx - 1]
                st.rerun()
        else:
            st.write("")

    with col_body:
        c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
        with c1:
            cfg['name'] = st.text_input(
                f"Layer {display_idx + 1} name",
                value=cfg['name'], key=f"name_{layer_id}",
            )
        with c2:
            is_last = display_idx == len(st.session_state.layer_order) - 1
            if is_last:
                semi_inf = st.checkbox(
                    "Semi-infinite",
                    value=cfg['thickness'] == np.inf,
                    key=f"semi_{layer_id}",
                    help="Opaque back contact (Au, Ag, Al). Uncheck for "
                         "transparent back / bifacial.",
                )
                if semi_inf:
                    st.text_input("Thickness (nm)", value="∞",
                                  disabled=True, key=f"th_dis_{layer_id}")
                    cfg['thickness'] = np.inf
                else:
                    cfg['thickness'] = st.number_input(
                        "Thickness (nm)", min_value=1.0, value=(
                            cfg['thickness'] if cfg['thickness'] != np.inf
                            else 100.0),
                        key=f"th_{layer_id}", format="%.1f",
                    )
            else:
                cfg['thickness'] = st.number_input(
                    "Thickness (nm)", min_value=1.0,
                    value=cfg['thickness'] if cfg['thickness'] != np.inf
                    else 100.0,
                    key=f"th_{layer_id}", format="%.1f",
                )
        with c3:
            cfg.setdefault('file_source', 'local')
            cfg.setdefault('local_file_name', '')
            src = st.radio(
                "Source:", ["Local file", "Upload"],
                index=0 if cfg['file_source'] == 'local' else 1,
                horizontal=True, key=f"src_{layer_id}",
                label_visibility="collapsed",
            )
            cfg['file_source'] = 'local' if src == "Local file" else 'upload'

            if src == "Local file" and nk_files:
                default_idx = (nk_files.index(cfg['local_file_name']) + 1
                               if cfg['local_file_name'] in nk_files else 0)
                sel = st.selectbox(
                    "n-k file", [""] + nk_files,
                    index=default_idx, key=f"sel_{layer_id}",
                )
                cfg['local_file_name'] = sel
                cfg['file'] = f"resources/nk data/{sel}" if sel else None
            else:
                cfg['file'] = st.file_uploader(
                    "n-k file", type=['txt', 'nk', 'dat'],
                    key=f"up_{layer_id}",
                )
                cfg['local_file_name'] = ''
        with c4:
            default_inc = (cfg['thickness'] > 1000 and
                           cfg['thickness'] != np.inf)
            inc = st.checkbox(
                "Incoherent",
                value=cfg['coherency'] == 'i' or default_inc,
                key=f"inc_{layer_id}",
                help="Check for thick layers (>1 µm) to remove interference "
                     "fringes.",
            )
            cfg['coherency'] = 'i' if inc else 'c'

    with col_down:
        if display_idx < len(st.session_state.layer_order) - 1:
            if st.button("⬇", key=f"dn_{layer_id}", help="Move down"):
                order = st.session_state.layer_order
                order[display_idx], order[display_idx + 1] = \
                    order[display_idx + 1], order[display_idx]
                st.rerun()
        else:
            st.write("")

    with col_del:
        if st.button("✕", key=f"del_{layer_id}", help="Delete layer"):
            st.session_state.layer_order.remove(layer_id)
            st.session_state.layer_configs.pop(layer_id, None)
            st.rerun()

    layers[display_idx] = cfg

# Active-layer selection
st.markdown("#### Active layer")
layer_names = [f"Layer {i+1}: {layers[i]['name']}" for i in range(len(layers))]
active_display_idx = st.selectbox(
    "Which layer is the absorber?",
    options=list(range(len(layers))),
    format_func=lambda x: layer_names[x],
    key="active_layer_selector",
)
active_layer_config = layers[active_display_idx]

# Wavelength / angle / resolution
st.markdown("#### Wavelength, angle, resolution")
wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
with wcol1:
    wl_min = st.number_input("λ min (nm)", 200, 1500, 350)
with wcol2:
    wl_max = st.number_input("λ max (nm)", 300, 2500, 800)
with wcol3:
    n_wavelengths = st.number_input("# λ points", 10, 500, 100)
with wcol4:
    angle_deg = st.slider("Incidence angle (°)", 0.0, 89.0, 0.0, 0.5,
                          help="0° = normal incidence.")
with wcol5:
    n_depths = st.number_input("# depth points", 50, 1000, 500, 50,
                               help="Resolution of G(z,λ) in the active layer.")

angle_rad = float(np.deg2rad(angle_deg))

# Spectrum
st.markdown("#### Incident spectrum")
spectrum_option = st.radio(
    "Spectrum", ["AM1.5G", "White light", "Upload custom"],
    horizontal=True,
)
custom_spectrum_file = None
if spectrum_option == "Upload custom":
    custom_spectrum_file = st.file_uploader(
        "Spectrum file (wavelength [nm], irradiance [W/m²/nm])",
        type=['sp', 'txt', 'dat'], key="custom_spec",
    )


# =============================================================================
# Shared helpers
# =============================================================================

def build_stack():
    """Prepend air; append air back if last layer is finite; return
    (materials, thicknesses, coherency, active_idx, is_semitransparent)."""
    materials = [lambda w: 1.0 + 0j]
    for lyr in layers:
        if lyr['file'] is None:
            return None, None, None, None, None
        materials.append(load_nk_file(lyr['file']))
    thicknesses = [np.inf] + [lyr['thickness'] for lyr in layers]
    coh = ['i'] + [lyr['coherency'] for lyr in layers]
    is_semitrans = thicknesses[-1] != np.inf
    if is_semitrans:
        materials.append(lambda w: 1.0 + 0j)
        thicknesses.append(np.inf)
        coh.append('i')
    coh[-1] = 'i'  # TMM requires incoherent outer layers
    active_idx = active_display_idx + 1  # +1 for front air
    return materials, thicknesses, coh, active_idx, is_semitrans


def resolve_spectrum(wavelengths: np.ndarray):
    """Return (flux_array, info_msg)."""
    if spectrum_option == "White light":
        return 1e15 * np.ones_like(wavelengths), "Flat white-light spectrum."
    if spectrum_option == "AM1.5G":
        path = "resources/Sunspectrum.sp"
        if os.path.exists(path):
            spec_wl, spec_flux = load_solar_spectrum(path)
            flux = interpolate_spectrum(spec_wl, spec_flux, wavelengths)
            return flux, (f"AM1.5G loaded from {path} "
                          f"({len(spec_wl)} pts, "
                          f"{spec_wl.min():.0f}–{spec_wl.max():.0f} nm).")
        return (2e15 * np.exp(-((wavelengths - 500) / 150) ** 2),
                f"AM1.5G file not found at {path} — using Gaussian fallback.")
    # Upload custom
    if custom_spectrum_file is None:
        return (1e15 * np.ones_like(wavelengths),
                "No file uploaded — using flat white-light fallback.")
    spec_wl, spec_flux = load_solar_spectrum(custom_spectrum_file)
    flux = interpolate_spectrum(spec_wl, spec_flux, wavelengths)
    return flux, (f"Custom spectrum loaded "
                  f"({len(spec_wl)} pts, "
                  f"{spec_wl.min():.0f}–{spec_wl.max():.0f} nm).")


# =============================================================================
# Tabs
# =============================================================================

tab_single, tab_sweep = st.tabs(["Single-point G(z,λ)", "Thickness sweep"])


# -----------------------------------------------------------------------------
# TAB 1 — Single-point G(z,λ)
# -----------------------------------------------------------------------------

with tab_single:
    st.markdown(
        "Compute R, T, per-layer absorptance, and the full spatially-resolved "
        "generation rate G(z,λ) in the active layer."
    )

    if st.button("Calculate generation profile", type="primary",
                 key="btn_single"):
        materials, thicknesses, coh, active_idx, is_semitrans = build_stack()
        if materials is None:
            st.error("Please provide n-k files for all layers.")
        else:
            wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)
            solar_flux, spec_msg = resolve_spectrum(wavelengths)
            st.info(spec_msg)
            if is_semitrans:
                st.info("Semi-transparent device: added air behind the last "
                        "layer.")
            if coh[active_idx] == 'i':
                st.warning("Active layer is incoherent — G(z,λ) will be zero.")

            with st.spinner("Running TMM…"):
                z_positions, generation = calculate_generation_tmm(
                    materials, thicknesses, coh, active_idx,
                    wavelengths, solar_flux, n_depths=n_depths,
                    angle_rad=angle_rad,
                )
                R, T, A = compute_rt_absorption(
                    materials, thicknesses, coh, wavelengths,
                    angle_rad=angle_rad,
                )

            absorptance_active = A[:, active_idx]
            total_generation = np.trapezoid(generation, wavelengths, axis=1)

            # 2x2 figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Total generation in {active_layer_config["name"]}',
                    'G(z,λ) heatmap',
                    f'Absorptance in {active_layer_config["name"]}',
                    'Absorption across layers',
                ),
                specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                       [{"type": "scatter"}, {"type": "scatter"}]],
                vertical_spacing=0.12, horizontal_spacing=0.10,
            )
            fig.add_trace(go.Scatter(
                x=z_positions, y=total_generation / 1e21, mode='lines',
                line=dict(color='rgb(220, 20, 60)', width=3),
                fill='tozeroy', fillcolor='rgba(220, 20, 60, 0.3)',
                showlegend=False,
            ), row=1, col=1)
            fig.update_xaxes(title_text="Depth in active layer (nm)",
                             row=1, col=1)
            fig.update_yaxes(title_text="G (×10²¹ cm⁻³ s⁻¹)",
                             row=1, col=1)

            fig.add_trace(go.Heatmap(
                x=wavelengths, y=z_positions, z=generation / 1e21,
                colorscale='Viridis',
                colorbar=dict(title="G(z,λ)<br>×10²¹", x=1.02, len=0.45, y=0.78),
                showscale=True,
            ), row=1, col=2)
            fig.update_xaxes(title_text="λ (nm)", row=1, col=2)
            fig.update_yaxes(title_text="Depth (nm)", row=1, col=2)

            fig.add_trace(go.Scatter(
                x=wavelengths, y=absorptance_active * 100, mode='lines',
                line=dict(color='rgb(50, 120, 180)', width=3),
                fill='tozeroy', fillcolor='rgba(50, 120, 180, 0.3)',
                showlegend=False,
            ), row=2, col=1)
            fig.update_xaxes(title_text="λ (nm)", row=2, col=1)
            fig.update_yaxes(title_text="Absorptance (%)", row=2, col=1,
                             range=[0, 100])

            # Per-layer absorption (skip front air; skip opaque back)
            palette = [
                'rgb(228,26,28)', 'rgb(55,126,184)', 'rgb(77,175,74)',
                'rgb(152,78,163)', 'rgb(255,127,0)', 'rgb(166,86,40)',
                'rgb(247,129,191)', 'rgb(153,153,153)', 'rgb(255,255,51)',
                'rgb(0,255,255)', 'rgb(128,0,128)', 'rgb(0,128,128)',
            ]
            trace_i = 0
            for i in range(len(materials)):
                if i == 0:
                    continue
                if i == len(materials) - 1 and thicknesses[-1] == np.inf:
                    if is_semitrans:
                        continue  # back air
                    else:
                        continue  # opaque back
                name = layers[i - 1]['name'] if (i - 1 < len(layers)) else f"L{i}"
                fig.add_trace(go.Scatter(
                    x=wavelengths, y=A[:, i] * 100, mode='lines',
                    line=dict(color=palette[trace_i % len(palette)], width=2),
                    name=name,
                ), row=2, col=2)
                trace_i += 1
            fig.add_trace(go.Scatter(
                x=wavelengths, y=R * 100, mode='lines',
                line=dict(color='black', width=2.5, dash='dash'),
                name='Reflection',
            ), row=2, col=2)
            if is_semitrans:
                fig.add_trace(go.Scatter(
                    x=wavelengths, y=T * 100, mode='lines',
                    line=dict(color='gray', width=2.5, dash='dot'),
                    name='Transmission',
                ), row=2, col=2)
            fig.update_xaxes(title_text="λ (nm)", row=2, col=2)
            fig.update_yaxes(title_text="Absorptance (%)", row=2, col=2)

            fig.update_layout(
                height=900, plot_bgcolor='white', paper_bgcolor='white',
                legend=dict(y=0.48, x=0.99, xanchor='right',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.3)', borderwidth=1,
                            font=dict(size=9)),
                margin=dict(l=70, r=120, t=60, b=70),
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

            # Download
            header = ("Charge generation [cm^-3*s^-1]\nColumn format:\n"
                      "x (nm)\t" +
                      "\t".join(f"{int(w)} nm" for w in wavelengths))
            buf = io.StringIO()
            np.savetxt(buf, np.column_stack([z_positions, generation]),
                       fmt='%.6e', delimiter='\t', header=header, comments='# ')
            st.download_button(
                "Download generation profile",
                data=buf.getvalue(),
                file_name="generation_profile_TMM.txt",
                mime="text/plain",
            )

            # Metrics
            jgen = jgen_from_absorptance(absorptance_active, solar_flux,
                                         wavelengths)
            peak_i = int(np.argmax(absorptance_active))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max G", f"{generation.max():.2e}")
            m2.metric("J_gen", f"{jgen:.2f} mA/cm²")
            m3.metric("Peak abs.", f"{absorptance_active[peak_i]*100:.1f}%",
                      help=f"at {wavelengths[peak_i]:.0f} nm")
            m4.metric("Band-avg R", f"{band_integrate(R, solar_flux, wavelengths)*100:.1f}%")


# -----------------------------------------------------------------------------
# TAB 2 — Thickness sweep
# -----------------------------------------------------------------------------

with tab_sweep:
    st.markdown(
        "Sweep the thickness of a single layer and plot band-integrated "
        "quantities (R, T, per-layer absorptance, J_gen in active layer) vs. "
        "the swept thickness. All other stack settings come from the panel "
        "above."
    )

    if not layers:
        st.info("Add layers in the stack definition above.")
    else:
        sw_col1, sw_col2 = st.columns([1, 2])
        with sw_col1:
            sweep_layer_display = st.selectbox(
                "Layer to sweep",
                options=list(range(len(layers))),
                format_func=lambda i: f"Layer {i+1}: {layers[i]['name']}",
                key="sweep_layer_display",
            )
        with sw_col2:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                t_min = st.number_input("Min (nm)", min_value=1.0, value=50.0,
                                        format="%.1f", key="sw_tmin")
            with r2:
                t_max = st.number_input("Max (nm)", min_value=1.0, value=800.0,
                                        format="%.1f", key="sw_tmax")
            with r3:
                n_pts = st.number_input("# points", 2, 50, 10, 1,
                                        key="sw_npts",
                                        help="Hard cap: 50.")
            with r4:
                log_spacing = st.checkbox("Log spacing", value=False,
                                          key="sw_log",
                                          help="Useful when sweeping across "
                                               "orders of magnitude.")

        if n_pts > 20:
            st.warning(f"{n_pts} sweep points may be slow — each point re-runs "
                       f"TMM over {n_wavelengths} wavelengths.")

        # Output selection
        output_choices = (
            ["Reflection (band-avg)"] +
            (["Transmission (band-avg)"] if layers[-1]['thickness'] != np.inf
             else []) +
            [f"Absorptance in {lyr['name']} (band-avg)" for lyr in layers] +
            ["J_gen in active layer (mA/cm²)"]
        )
        default_outputs = [
            "Reflection (band-avg)",
            f"Absorptance in {active_layer_config['name']} (band-avg)",
            "J_gen in active layer (mA/cm²)",
        ]
        default_outputs = [o for o in default_outputs if o in output_choices]
        selected_outputs = st.multiselect(
            "Quantities to plot vs. thickness",
            output_choices, default=default_outputs,
        )

        vs_lambda_choice = st.selectbox(
            "Also plot vs. wavelength (one curve per sweep point)",
            ["None", "R(λ)", "Transmission T(λ)",
             *[f"Absorptance A(λ) in {lyr['name']}" for lyr in layers]],
            index=0,
        )

        if st.button("Run thickness sweep", type="primary", key="btn_sweep"):
            if any(lyr['file'] is None for lyr in layers):
                st.error("Please provide n-k files for all layers.")
            elif t_min >= t_max:
                st.error("Min thickness must be smaller than Max.")
            else:
                materials, thicknesses, coh, active_idx, is_semitrans = \
                    build_stack()
                if log_spacing:
                    sweep_values = np.logspace(np.log10(t_min),
                                               np.log10(t_max), int(n_pts))
                else:
                    sweep_values = np.linspace(t_min, t_max, int(n_pts))

                wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)
                solar_flux, spec_msg = resolve_spectrum(wavelengths)
                st.info(spec_msg)

                # Index of swept layer inside the padded thickness list
                swept_idx = sweep_layer_display + 1
                nL = len(materials)

                # Containers for sweep results
                R_arr = np.zeros((len(sweep_values), len(wavelengths)))
                T_arr = np.zeros_like(R_arr)
                A_arr = np.zeros((len(sweep_values), len(wavelengths), nL))

                progress = st.progress(0.0, text="Running TMM sweep…")
                t0 = time.time()
                for k, tv in enumerate(sweep_values):
                    th = list(thicknesses)
                    th[swept_idx] = float(tv)
                    R, T, A = compute_rt_absorption(
                        materials, th, coh, wavelengths, angle_rad=angle_rad,
                    )
                    R_arr[k] = R
                    T_arr[k] = T
                    A_arr[k] = A
                    progress.progress((k + 1) / len(sweep_values),
                                      text=f"{k+1}/{len(sweep_values)} pts "
                                           f"({time.time()-t0:.1f} s)")
                progress.empty()
                st.success(f"Sweep done in {time.time()-t0:.1f} s.")

                # Build per-output band-integrated series
                series = {}
                for out in selected_outputs:
                    if out == "Reflection (band-avg)":
                        series[out] = np.array([
                            band_integrate(R_arr[k], solar_flux, wavelengths)
                            for k in range(len(sweep_values))
                        ]) * 100
                    elif out == "Transmission (band-avg)":
                        series[out] = np.array([
                            band_integrate(T_arr[k], solar_flux, wavelengths)
                            for k in range(len(sweep_values))
                        ]) * 100
                    elif out.startswith("Absorptance in "):
                        name = out[len("Absorptance in "):].rsplit(
                            " (band-avg)", 1)[0]
                        lyr_display = next(i for i, lyr in enumerate(layers)
                                           if lyr['name'] == name)
                        li = lyr_display + 1
                        series[out] = np.array([
                            band_integrate(A_arr[k, :, li], solar_flux,
                                           wavelengths)
                            for k in range(len(sweep_values))
                        ]) * 100
                    elif out.startswith("J_gen"):
                        series[out] = np.array([
                            jgen_from_absorptance(A_arr[k, :, active_idx],
                                                   solar_flux, wavelengths)
                            for k in range(len(sweep_values))
                        ])

                # Plot vs thickness
                has_jgen = any(k.startswith("J_gen") for k in series)
                has_pct = any(not k.startswith("J_gen") for k in series)

                fig1 = make_subplots(specs=[[{"secondary_y": has_jgen and
                                               has_pct}]])
                palette = [
                    'rgb(228,26,28)', 'rgb(55,126,184)', 'rgb(77,175,74)',
                    'rgb(152,78,163)', 'rgb(255,127,0)', 'rgb(166,86,40)',
                    'rgb(247,129,191)', 'rgb(153,153,153)',
                ]
                for i, (label, y) in enumerate(series.items()):
                    is_jgen = label.startswith("J_gen")
                    fig1.add_trace(
                        go.Scatter(
                            x=sweep_values, y=y, mode='lines+markers',
                            name=label,
                            line=dict(color=palette[i % len(palette)], width=2),
                        ),
                        secondary_y=is_jgen and has_pct,
                    )
                fig1.update_xaxes(
                    title_text=f"Thickness of {layers[sweep_layer_display]['name']} (nm)",
                    type="log" if log_spacing else "linear",
                )
                if has_pct:
                    fig1.update_yaxes(title_text="R / T / Absorptance (%)",
                                       secondary_y=False)
                if has_jgen:
                    fig1.update_yaxes(title_text="J_gen (mA/cm²)",
                                       secondary_y=has_pct)
                fig1.update_layout(
                    height=500, plot_bgcolor='white', paper_bgcolor='white',
                    legend=dict(x=0.02, y=0.98,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='rgba(0,0,0,0.3)', borderwidth=1),
                    margin=dict(l=70, r=70, t=40, b=60),
                )
                st.plotly_chart(fig1, use_container_width=True, theme=None)

                # Download CSV
                csv_header = ["thickness_nm"] + list(series.keys())
                csv_data = np.column_stack(
                    [sweep_values] + [series[k] for k in series]
                )
                buf = io.StringIO()
                buf.write(",".join(csv_header) + "\n")
                np.savetxt(buf, csv_data, delimiter=",", fmt="%.6e")
                st.download_button(
                    "Download sweep data (CSV)",
                    data=buf.getvalue(),
                    file_name=f"sweep_{layers[sweep_layer_display]['name']}.csv",
                    mime="text/csv",
                )

                # Optional vs-λ plot
                if vs_lambda_choice != "None":
                    if vs_lambda_choice == "R(λ)":
                        Y = R_arr
                        yl = "Reflection (%)"
                    elif vs_lambda_choice == "Transmission T(λ)":
                        Y = T_arr
                        yl = "Transmission (%)"
                    else:
                        name = vs_lambda_choice[len("Absorptance A(λ) in "):]
                        lyr_display = next(i for i, lyr in enumerate(layers)
                                           if lyr['name'] == name)
                        li = lyr_display + 1
                        Y = A_arr[:, :, li]
                        yl = f"Absorptance in {name} (%)"

                    fig2 = go.Figure()
                    # colormap by thickness (viridis-like)
                    cmin, cmax = float(sweep_values.min()), float(sweep_values.max())
                    for k, tv in enumerate(sweep_values):
                        frac = ((np.log10(tv) - np.log10(cmin)) /
                                (np.log10(cmax) - np.log10(cmin))
                                if log_spacing and cmax > cmin
                                else (tv - cmin) / (cmax - cmin) if cmax > cmin
                                else 0.0)
                        r_val = int(68 + frac * (253 - 68))
                        g_val = int(1 + frac * (231 - 1))
                        b_val = int(84 + frac * (37 - 84))
                        fig2.add_trace(go.Scatter(
                            x=wavelengths, y=Y[k] * 100, mode='lines',
                            line=dict(color=f"rgb({r_val},{g_val},{b_val})",
                                      width=1.5),
                            name=f"{tv:.1f} nm",
                        ))
                    fig2.update_layout(
                        title=vs_lambda_choice,
                        xaxis_title="λ (nm)",
                        yaxis_title=yl,
                        height=500, plot_bgcolor='white',
                        paper_bgcolor='white',
                        legend=dict(font=dict(size=9)),
                        margin=dict(l=70, r=40, t=50, b=60),
                    )
                    st.plotly_chart(fig2, use_container_width=True, theme=None)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(
    "TMM Simulator · Novel Semiconductor Devices Team · "
    "Uses the [`tmm`](https://github.com/sbyrnes321/tmm) package. "
    "Bundled perovskite n-k data from the "
    "[Dasgupta repository](https://github.com/akashdasgupta/Perovskite-Dielectric-Constants-Repository)."
)
