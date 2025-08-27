# Architecture

- **MVC-ish**: controllers (UI adapters), views (Streamlit components), services (domain logic),
  providers (data), models (ML), features (engineering).
- **Interfaces**: files are prefixed `i_` but class names don't start with `I`.
- **Frontends**: Streamlit (`app.py`) and console (`main.py`) share the same backend.
- **Config**: simple YAML in `configs/`; easily swappable for Hydra/Pydantic Settings.