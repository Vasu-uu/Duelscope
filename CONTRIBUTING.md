# Contributing to Duelscope

Thanks for helping! The short version:

```bash
git clone https://github.com/Vasu-uu/Duelscope.git
cd Duelscope
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q && ruff check .
```

## Guidelines

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`).
- Keep modules small, add type hints and docstrings on public functions.
- Every metric or selection change needs a test with a hand-computed expected value.
- Never commit datasets, model weights, or generated reports.
- Never put invented benchmark numbers in docs; numbers must come from a real run on stated hardware.
- New detector? Start from `examples/custom_adapter.py` and the "new detector adapter" issue template.
