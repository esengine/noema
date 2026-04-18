# Noema project conventions

Read this before editing any file in this repo. Applies to humans and AI assistants equally.

## Comments

**Default: write none.** Code is the source of truth; well-named identifiers explain *what*. Only add a comment when the *why* is non-obvious and would surprise a future reader. Examples of valid reasons:

- A hidden constraint imposed by an external system (`vocab_size = 50304  # GPU tile size`).
- A subtle invariant the type system can't express.
- A workaround for a specific upstream bug, with a link.
- A non-obvious numerical or training trick the reader can't infer from the code.

Forbidden comment patterns:

- Section headers inside dataclasses or functions (`# Data`, `# Optim`, `# Gradient accumulation`).
- Narrating the next line (`# load the model`, `# first pass: count tokens`).
- Restating the function/variable name in prose.
- User-facing tips, "you can flip this on", "feel free to change". These belong in `README.md` or `docs/`.
- TODO comments without an issue link or a date.
- Decorative dividers (`# ===== section =====`).

If you find yourself writing a comment to make code understandable, **rename the identifier or extract a function instead**.

## Docstrings

- No module-level docstrings. The filename and contents are enough.
- No class docstrings unless the class has non-trivial invariants.
- Function docstrings only when the signature genuinely doesn't convey intent. One short line, no `Args:`/`Returns:` sections — types do that.
- Never write multi-paragraph docstrings.

## File / module style

- One concept per file. If a file grows past ~300 lines, consider splitting.
- Type-annotate public function signatures. Internal helpers may skip annotations if obvious.
- `from __future__ import annotations` at the top of every Python file.
- No emojis anywhere — code, comments, commits, docs.

## Naming

- Modules: lowercase, no underscores when avoidable (`model.py`, not `gpt_model.py`).
- Classes: `PascalCase`. Functions / variables: `snake_case`. Constants: `UPPER_SNAKE`.
- Avoid abbreviations except universal ones (`cfg`, `idx`, `id`, `lr`, `attn`).

## Configs

- Configs in `configs/` may have a single header line stating the config's purpose.
- Per-field comments only for non-obvious values (a citation, a hardware constraint).
- Never duplicate the field name in a comment.

## Tests

- Tests live in `tests/`, mirror `src/noema/` layout.
- One behavior per test. No test docstrings — name the test for the behavior.
- Use `pytest`, no `unittest`.

## Documentation vs. code

- *How to use the project*: `README.md`.
- *Why we made a research choice*: `docs/research_plan.md` or a new file in `docs/`.
- *What a function does*: the function's name and body.

If you're tempted to put usage instructions in a Python file's docstring, put them in `README.md` and link to the script instead.

## Commits

- Imperative mood, lowercase, no trailing period: `add latent thinking head`, not `Added the latent thinking head.`
- Subject ≤ 70 chars. Body wraps at 80 if present.
- One logical change per commit.
- No "fix typo", "wip", "update" commits on `main` — squash before merging.

## Pull requests

- Title mirrors the commit subject style.
- Body: what changed, why, how to verify. No screenshots unless visual.
- Link the issue or research-plan section being addressed.

## Dependencies

- Add to `pyproject.toml`, never `pip install` ad-hoc into a shared env.
- Justify new dependencies in the PR body.
- Pin lower bounds only (`>=`); don't pin upper bounds without a known incompatibility.

## What this repo is not

- Not a chatbot product.
- Not a place for prompt-engineering tricks or LLM wrapper code.
- Not a benchmark farm — we run small, focused evals tied to research questions.

If a change doesn't move us toward answering a research question in `docs/research_plan.md`, it probably doesn't belong here.
