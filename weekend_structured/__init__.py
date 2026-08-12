"""The structured-output grid.

A package rather than a bare script directory, unlike days 1-6, and not by
preference. `day3_sampling/config.py` and this directory's `config.py` are both
inserted onto `sys.path` by pytest, so `import config` resolved to whichever
directory was collected first — day 3's — and the whole weekend failed to import
under `uv run pytest` while passing when run alone.

pyproject.toml predicted the general case on day 5 ("importing each other by
accident of sys.path[0] ... stops working the moment two days need to share
code"). This is the other half of it: two days that share nothing but a *filename*
collide just as hard. Being a package makes `weekend_structured.config`
unambiguous, at the cost of `-m` invocation:

    uv run python -m weekend_structured.run
"""
