# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-12

### Added
- `streamlit_app.py` — interactive 4-tab Streamlit demo: Architecture, Latency Budget, Circuit Breaker, API & Config
- `demo_bundle.py` — standalone script that simulates 500 pipeline turns and produces `data/demo_bundle.pkl` (no API keys or ML models required)
- `data/demo_bundle.pkl` — pre-computed Streamlit demo bundle (latency distributions, circuit-breaker events, intent counts, stage stats)
- `requirements-ci.txt` — lean CI dependency file (FastAPI, pytest, asyncio utilities; excludes faster-whisper, librosa, opencv, edge-tts, openai)
- `.streamlit/config.toml` — dark purple theme
- `runtime.txt` — Streamlit Cloud Python version pin (`python-3.11`)
- `packages.txt` — system packages for Streamlit Cloud (`ffmpeg`, `libsndfile1`)

### Changed
- `.github/workflows/ci.yml`:
  - `test` job now uses `requirements-ci.txt` (faster install, no CUDA deps)
  - `codecov/codecov-action` upgraded `@v4` → `@v5`
  - Retained Python 3.11 / 3.12 matrix, lint job, and Docker smoke-test job
- `requirements.txt` — added `streamlit>=1.36.0` and `plotly>=5.18.0`
- `tests/integration/test_api.py` — replaced real lifespan with async mock context manager so integration tests pass without loading faster-whisper; all 76 tests now pass
- `README.md` — added Streamlit Demo section, updated Table of Contents, updated Project Structure tree

### Fixed
- 12 integration tests that were failing because `TestClient` lifespan tried to load `faster_whisper.WhisperModel` in CI environments without the binary
