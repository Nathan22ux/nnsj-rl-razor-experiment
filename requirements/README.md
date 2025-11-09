requirements folder

This folder contains example dependency files for the project.

Files
- base.txt: runtime dependencies required to run the project.
- dev.txt: development dependencies (tests, linters, formatters, notebooks).

Commands

# Export your current environment into base.txt (run from the virtualenv/interpreter you want to capture):
python -m pip freeze > requirements/base.txt

# Install runtime dependencies from base.txt:
python -m pip install -r requirements/base.txt

# Install development dependencies as well:
python -m pip install -r requirements/dev.txt

Notes
- Prefer using `python -m pip` to ensure pip runs under the correct interpreter.
- If you want stricter dependency management, consider using pip-tools (pip-compile / pip-sync) or poetry.
- Pin important packages before sharing or deploying to avoid surprises.
