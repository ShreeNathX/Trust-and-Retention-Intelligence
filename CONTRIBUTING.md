# Contributing

Issues and pull requests are welcome.

## Before opening a pull request

- Describe the change and the reason for it.
- Keep model inputs, target definition, and public-data limitations explicit.
- Never include private customer data, credentials, environment folders, or notebook checkpoints.
- For modeling changes, run `Model.ipynb` from top to bottom, check that the app loads the saved artifact, and update the benchmark in `README.md` from the new run.
- Keep model selection on training cross-validation. Do not choose features, models, or thresholds from the locked holdout.
- If a change affects the app, include the steps used to verify the app and example inputs.

## Local workflow

1. Install notebook and development dependencies with `python -m pip install -r requirements-dev.txt`.
2. Run every cell in `Model.ipynb` from the project folder.
3. Run `python -m unittest discover -s tests -v` to check the saved model and reports.
4. Start the app with `streamlit run app.py` and confirm the scoring and benchmark tabs load.
5. Summarize changes and any benchmark movement in the pull request.

The public dataset has no observation date or future churn horizon. Contributions must not present its scores as validated future-churn or bank-specific performance.
