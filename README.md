# TMM Simulator

Transfer-matrix optical simulations for thin-film solar stacks, with
single-point G(z,λ) analysis and layer-thickness sweeps. Streamlit UI.

- **Single-point tab**: compute spatially-resolved generation rate G(z,λ) in
  the active layer, per-layer absorptance, reflection and transmission.
- **Thickness sweep tab**: pick any layer, scan a range of thicknesses, and
  plot band-integrated R, T, per-layer absorptance, and J_gen in the active
  layer vs. the swept thickness. Linear or log spacing. Optional "vs. λ" view
  overlays each sweep point colored by thickness.
- Incoherent layers supported (thick substrates like glass).
- Off-normal incidence supported via angle slider (0–89°).
- Bundled n-k library:
  - Generic materials (glass, FTO, ITO, Au, PEDOT, TiO₂, C60, MgF₂, …) under
    `resources/nk data/`.
  - Perovskite dielectric constants from the
    [Dasgupta repository](https://github.com/akashdasgupta/Perovskite-Dielectric-Constants-Repository)
    under `resources/nk data/dasgupta/`.
- NREL AM1.5G spectrum shipped as `resources/Sunspectrum.sp`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to <https://share.streamlit.io/> and sign in with GitHub.
3. **New app** → pick this repo and branch → main file: `app.py`.
4. Streamlit Cloud will install `requirements.txt` and launch the app.

## File layout

```
tmm-app/
├── app.py                  Streamlit UI (two tabs on top of a shared stack-definition panel)
├── tmm_core.py             Pure-Python TMM helpers (no Streamlit imports)
├── requirements.txt        Pinned deps for reproducible deploys
├── .streamlit/config.toml  Light theme, 50 MB upload cap
├── resources/
│   ├── Sunspectrum.sp      AM1.5G irradiance (W/m²/nm)
│   └── nk data/            n-k files; subfolders can be flat or nested
├── LICENSE                 MIT
└── README.md
```

## Credits

- Transfer-matrix engine: the [`tmm`](https://github.com/sbyrnes321/tmm)
  package by Steven Byrnes.
- Perovskite n-k data: the
  [Perovskite Dielectric Constants Repository](https://github.com/akashdasgupta/Perovskite-Dielectric-Constants-Repository)
  by Akash Dasgupta et al.

## License

MIT — see [`LICENSE`](LICENSE). Novel Semiconductor Devices Team.
