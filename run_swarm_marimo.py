import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electride discovery with a reasoning swarm

    Open this notebook from the repository folder in Jupyter with a Python 3.11+ kernel, then choose **Run All**. The first cell installs dependencies. Section 1 asks you to choose your LLM provider and enter your API key in a masked field; the remaining cells run automatically.

    The sample dataset, feature catalog, configuration, and local papers are included. Settings: `config.yaml`. Code: `src/agent.py` (schemas, prompts, swarm loop), `src/tools.py` (web, calculator, arXiv, local RAG), and `src/evaluator.py` (deterministic evaluation). Add extra PDFs to `inputs/kb/` if desired.
    """)
    return


@app.cell
def _():
    # '%pip install -r src/requirements.txt' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choose your LLM provider

    Choose **OpenRouter**, **OpenAI**, or **Other / custom endpoint** below. OpenRouter and OpenAI have preset endpoints and models, so you only need your API key. For a custom endpoint, also enter its API base URL and model ID. It must support OpenAI-compatible Chat Completions with JSON mode and `max_tokens`.

    Press Enter after each answer. Your key is masked and is not written into the notebook source or outputs. No environment variables or code edits are required.
    """)
    return


@app.cell
def _():
    import functools
    import os
    from getpass import getpass

    import instructor
    from openai import AsyncOpenAI

    print("Choose your LLM provider:")
    print("  1. OpenRouter")
    print("  2. OpenAI")
    print("  3. ORNL SambaNova")
    print("  4. Other / custom endpoint (OpenAI-compatible)")
    choice = input("Provider [1–4]: ").strip()
    while choice not in {"1", "2", "3", "4"}:
        choice = input("Please enter 1, 2, 3, or 4: ").strip()

    presets = {
        "1": ("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-oss-120b"),
        "2": ("OpenAI", "https://api.openai.com/v1", "gpt-5.6-terra"),
        "3": ("ORNL SambaNova", "https://sambanova.cades.ornl.gov/v1", "gpt-oss-120b"),
    }
    if choice in presets:
        provider, base_url, MODEL_NAME = presets[choice]
    else:
        provider = "Custom endpoint"
        base_url = input("API base URL: ").strip()
        MODEL_NAME = input("Model name: ").strip()
        while not base_url or not MODEL_NAME:
            print("The custom option needs both an API URL and a model name.")
            base_url = input("API base URL: ").strip()
            MODEL_NAME = input("Model name: ").strip()

    print(f"\n{provider} · {MODEL_NAME}")
    _api_key = os.environ.get("SN_API_KEY", "").strip() if choice == "3" else ""
    if _api_key:
        print("API key: taken from the SN_API_KEY environment variable")
    else:
        _api_key = getpass("API key: ").strip()
    while not _api_key:
        _api_key = getpass("Please enter your API key: ").strip()
    client = AsyncOpenAI(base_url=base_url, api_key=_api_key)
    del _api_key
    async_gen = instructor.from_openai(client, mode=instructor.Mode.JSON)

    # Preserve OpenRouter's and ORNL's sampling settings; other providers use their defaults.
    TEMPERATURE = 0.7 if choice in {"1", "3"} else None
    call_settings = {"max_retries": 3}
    if choice == "2":
        call_settings["max_completion_tokens"] = 8192
    else:
        call_settings["max_tokens"] = 8192
    if TEMPERATURE is not None:
        call_settings["temperature"] = TEMPERATURE
    _create = functools.partial(async_gen.chat.completions.create, **call_settings)
    N_TRIES = 5

    async def _create_with_retry(**kwargs):
        for attempt in range(N_TRIES):
            try:
                call_kwargs = dict(kwargs)
                if attempt and TEMPERATURE is not None:
                    call_kwargs["temperature"] = max(0.1, TEMPERATURE - 0.15 * attempt)
                return await _create(**call_kwargs)
            except Exception as exc:
                if attempt == N_TRIES - 1:
                    raise
                print(f"LLM call failed (attempt {attempt + 1}/{N_TRIES}: {type(exc).__name__}); retrying")

    async_gen.chat.completions.create = _create_with_retry
    print("Setup complete. Continue with the swarm below.")
    return MODEL_NAME, async_gen


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Config
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    import sys; sys.path.insert(0, "src")
    from agent import Swarm, load_config
    from evaluator import score_dataset

    CONFIG = load_config("config.yaml")
    return CONFIG, Swarm, score_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Build the swarm

    Uses the LLM from section 1; loads the embedding model, the feature data + evaluator, and the tools (builds the local paper index).
    """)
    return


@app.cell
def _(CONFIG, MODEL_NAME, Swarm, async_gen):
    swarm = Swarm(CONFIG, llm=async_gen, model_name=MODEL_NAME)
    swarm.plot_graph()
    return (swarm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Run

    Every round each agent may use one tool, then returns a `proposal` (the feature set its model should use). A gradient-boosting classifier is fitted on exactly those features and evaluated on all labeled structures over repeated stratified splits; the evaluation opens its next round. After the last round the top-ranked feature set is refit on all structures and the model is saved to `outputs/final_model_*.joblib`.
    """)
    return


@app.cell
async def _(swarm):
    last_round, results = await swarm.run()
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Analysis of the run

    All plots come from the evaluations the swarm already did (`swarm.evaluator.history`), so nothing is refitted.
    """)
    return


@app.cell
def _(swarm):
    import analysis

    # How each agent's score evolves, and the swarm's best-so-far
    progress = analysis.plot_agent_progress(swarm)
    progress
    return (analysis,)


@app.cell
def _(analysis, swarm):
    # Which features each agent selected in each round (% = share of that round's fitted model decisions)
    analysis.plot_feature_map(swarm)
    return


@app.cell
def _(analysis, swarm):
    # How often each feature was selected per round, across all agents
    analysis.plot_feature_usage(swarm)
    return


@app.cell
def _(analysis, results, swarm):
    # Classification metrics of the top-ranked feature set (validation predictions pooled over the splits)
    metrics = analysis.plot_final_model(swarm, results)
    metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Structure viewer

    Pass the CIF paths of one electride and one non-electride. `pick_example_structures` finds the most confidently predicted example of each class and its CIF under `cif_dir`; otherwise give the two paths yourself. Use `repeat=(2, 2, 1)` to tile the cell and `rotations=` for other orientations.
    """)
    return


@app.cell
def _(analysis):
    # Option A: let the top model pick the clearest example of each class (needs CIFs in cif_dir)
    # el_cif, non_cif, probability = analysis.pick_example_structures(swarm, results, cif_dir="structures")

    # Option B: your own two files
    el_cif, non_cif, probability = "data/data_extract/structures/mp-11854.cif", "data/data_extract/structures/mp-559931.cif", None

    analysis.show_structures(el_cif, non_cif, repeat=(2, 2, 1), probability=probability)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Score a dataset with the final model

    Swap `DATA_CSV` for your separate dataset (labels optional). In a new session use `score_dataset(None, ...)` for the newest `outputs/final_model_*.joblib`, or pass its path.
    """)
    return


@app.cell
def _(CONFIG, results, score_dataset):
    DATA_CSV = CONFIG["data"]["dataset_csv"]
    scores, score_metrics = score_dataset(results["final_fit"], DATA_CSV, out_csv="outputs/final_model_scores.csv")
    scores.sort_values("probability", ascending=False)
    return


if __name__ == "__main__":
    app.run()
