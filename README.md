# Electride Discovery with a Reasoning Swarm

A multi-agent system in which several LLM scientists independently reason about **why** a material is an
electride, propose which physical descriptors a classifier should be built from, and refine those proposals
over several rounds using deterministic, per-class evidence from the data.

Electrides are ionic compounds in which some valence electrons are not bound to any atom: they occupy
interstitial cavities, channels or layers and behave as anions. Deciding which measurable structural and
chemical quantities reveal that behaviour is a scientific judgement, not a fitting problem. This project puts
the judgement in the hands of reasoning agents and leaves the arithmetic to a fixed, auditable backend.

---

## What this is, and what it is not

| | |
|---|---|
| **The agents decide** | which physical quantities the model may use, and why; what the literature supports; what to change after seeing which class fails |
| **The backend decides** | nothing about the physics. It fits a gradient-boosting classifier on exactly the features it is given and reports honest validation numbers |
| **Not an AutoML loop** | no feature search, no hyperparameter search, no model selection by the backend. Every candidate comes from an agent with a stated physical reason |
| **Not a black box** | each proposal is a short list of named physical descriptors, each with an expected direction. The run log records every proposal, its evidence and its score |

---

## How a round works

```
             ┌──────────── system prompt: problem, rules, usable feature catalog ────────────┐
             │                                                                               │
  Round r    ▼                                                                               │
  ┌──────────────────┐   one tool per round    ┌───────────────────────────────┐             │
  │  Agent A1 … A5   │ ──────────────────────► │ literature_search / fetch_    │             │
  │  (independent)   │ ◄────────────────────── │ paper / kb_search / web /calc │             │
  └────────┬─────────┘      real evidence      └───────────────────────────────┘             │
           │ proposal: FeatureSet (features + physical role + hypothesis + reasoning)         │
           ▼                                                                                 │
  ┌────────────────────────────────────────────────────────────────────────────┐             │
  │ Deterministic evaluation (evaluator.py)                                    │             │
  │  • gradient boosting fitted on exactly those features                      │             │
  │  • 5 stratified train/validation splits of ALL labeled structures          │             │
  │  • per-class metrics, per-feature role, hardest structures, parent compare │             │
  └────────┬───────────────────────────────────────────────────────────────────┘             │
           │ feedback (plain words + numbers)                                                 │
           ▼                                                                                 │
  Round r+1: each agent sees its own lineage + feedback, and 2 diverse peers' proposals ──────┘
```

After the last round the top-ranked feature set is refit on all labeled structures and exported for scoring
new data.

### Why repeated splits

Each proposal is scored on 5 stratified 75/25 splits with fixed seeds. The coefficients (or, here, the trees)
are fitted on the training part and scored on the held-out part, so a proposal cannot win by memorising the
data. Scores are reported as mean ± standard deviation, and the agents are told explicitly when a change is
inside that noise band.

### What an agent actually reads

Each round's feedback contains, in plain language first:

* **Ranking quality** — ROC-AUC with an interpretation ("separates them moderately well").
* **Which class fails** — e.g. "Main failure: CALLING NON-ELECTRIDES ELECTRIDES. It finds 90% of electrides but
  rejects only 67% of non-electrides."
* **Each feature, by class** — its share of the model's decisions; its class means; what it achieves *used alone*
  as a threshold rule; how much electride recall vs non-electride rejection drops when it is scrambled
  ("mainly needed to DETECT electrides" / "mainly needed to REJECT non-electrides" / "little effect").
* **Redundancy** — strongly correlated feature pairs.
* **Cutoff trade-offs** — "to find 90% of electrides, the model would reject 41% of non-electrides".
* **The hardest structures** — the non-electrides scored most electride-like and vice versa, by formula.
* **Parent comparison** — paired over the same splits: "LOWER than its parent #7 … lower in 5 of 5 splits."

The feedback states facts only. What to do about them is the agent's decision, guided by the system prompt.

---

## Repository layout

```
.
├── run_swarm.ipynb                    # the only notebook: LLM setup → config → build → run → analysis
├── config.yaml                        # every structural setting (no secrets, no LLM endpoint)
├── agent.py                           # schemas, prompts, peer selection, agent round, swarm loop
├── evaluator.py                       # deterministic evaluation, ranking, export, score_dataset
├── tools.py                           # web_search, calculator, arXiv search/download, local-paper RAG
├── analysis.py                        # post-run plots (progress, feature maps, metrics, structure viewer)
├── requirements.txt
├── electride_feature_rag_clean.jsonl  # feature catalog: function, feature, description  (your file)
├── electride_features_dataset_200.csv # precomputed features + labels                    (your file)
├── inputs/kb/                         # local PDFs + papers agents download
└── outputs/                           # run logs, exported models, scores
```

### Responsibilities

* **`agent.py`** — the `FeatureSet` / `ScientificAgent` schemas, the system prompt, the problem statement, the
  communication graph, diverse peer selection, the per-round loop, the final ranking and export, and the
  tables/plots printed during a run. Receives a ready LLM client from the notebook.
* **`evaluator.py`** — `FeatureSetEvaluator` (what the swarm calls), `evaluate_feature_set` (one fit + one
  split), and `score_dataset` (apply an exported model to any CSV). No LLM calls anywhere in this file.
* **`tools.py`** — one function per tool, each taking a single string and always returning a string, so an agent
  always sees either evidence or an explicit failure message.
* **`analysis.py`** — read-only plots over `swarm.evaluator.history`; nothing is refitted.

---

## Input data

### 1. Feature catalog (`*.jsonl`)

One JSON record per line:

```json
{"function": "cavity_geometry", "feature": "max_clearance_A", "description": "Distance from the void centre to the nearest atom's covalent surface, in Å."}
```

The catalog is what the agents see. Descriptions matter: they are the only thing telling an agent what a
feature physically means.

### 2. Feature table (`*.csv`)

One row per structure:

| column | meaning |
|---|---|
| `structure` | identifier (e.g. `matpes-2024_1021_0.cif`) |
| `label` | `1` = electride, `0` = non-electride |
| `<function>__<feature>` | one column per catalog entry, e.g. `cavity_geometry__max_clearance_A` |

`true`/`false` strings are converted to 1/0; missing values are imputed with the **training fold's** median.

A catalog feature is offered to the agents only if its column exists, is numeric, varies, and is present in at
least `min_coverage` of rows. JSON-valued columns (lists, nested records) and constant run settings are listed
to the agents as *not usable*, so they can request numeric aggregations of them instead.

### 3. Local papers (`inputs/kb/`)

Drop PDFs in here. They are chunked, embedded with `all-MiniLM-L6-v2` and searchable with `kb_search`.
Papers that agents download with `fetch_paper` land in the same folder and persist across runs. Each round the
agents are told what the collection currently contains, so they can target one paper:
`kb_search("[Assessing the design rules of electrides.pdf] mean clearance")`.
Scanned PDFs with no text layer are reported at index time ("no extractable text … run OCR first").

---

## Installation

Python **3.11+** (required by the arXiv MCP server).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face.

### LLM access

The endpoint, model and key live in **section 1 of the notebook**, because they change from machine to machine:

```python
MODEL_NAME = "gpt-oss-120b"
client = AsyncOpenAI(base_url="https://your-endpoint/v1", api_key=os.environ["SN_API_KEY"])
async_gen = instructor.from_openai(client, mode=instructor.Mode.JSON)
```

Any OpenAI-compatible endpoint works. `instructor` enforces the response schema; the notebook adds a
multi-attempt retry whose temperature steps **down** on each retry (lower temperature produces valid JSON more
reliably).

> If the host name cannot be resolved (`nodename nor servname provided`), the problem is network/VPN, not the
> code — the request never leaves the machine.

---

## Running

```python
swarm = Swarm(CONFIG, llm=async_gen, model_name=MODEL_NAME)   # builds evaluator, embedder, tools, KB index
swarm.plot_graph()
last_round, results = await swarm.run()
```

Each round prints two tables (tools and evidence used, features selected, per-class scores, notes; and the
agents' reasoning with their suggested new features), followed by a ROC curve per agent with the round's best
model drawn bold. At the end you get the ranking, the exported top model, a summary of every new feature the
agents asked for, and token usage per agent.

`results` contains `leaderboard`, `selected`, `final_fit`, `suggested_features` and the full `history`.

### Outputs

| file | contents |
|---|---|
| `outputs/log_run_<timestamp>.jsonl` | every round: prompts, tool calls, full tool results, proposals, evaluations, token usage, final ranking |
| `outputs/final_model_<timestamp>.joblib` | the exported model: estimator, imputer, column order, training medians, feature names |
| `outputs/final_model_scores.csv` | per-structure probabilities from `score_dataset` |
| `outputs/arxiv_mcp.log` | arXiv MCP server stderr |

### Scoring new structures

```python
from evaluator import score_dataset
scores, metrics = score_dataset(results["final_fit"], "my_new_structures.csv")
# later session, newest exported model:
scores, metrics = score_dataset(None, "my_new_structures.csv")
```

The new CSV needs only the feature columns the model uses. `label` is optional; when present you also get
metrics and a confusion matrix. The same training medians are reused for missing values.

---

## Analysis plots (`analysis.py`)

```python
import analysis
progress = analysis.plot_agent_progress(swarm)   # score per agent per round + swarm best-so-far
analysis.plot_feature_map(swarm)                 # per agent: which features, which round, how much used
analysis.plot_feature_usage(swarm)               # how many agents picked each feature, per round
metrics = analysis.plot_final_model(swarm, results)  # ROC, PR, confusion matrix, score separation
analysis.show_structures("structures/Ca2N.cif", "structures/NaCl.cif", repeat=(2, 2, 1))
```

`plot_feature_map` is the clearest view of the search: each agent's features down the side, rounds across, the
cell shaded by the share of the model's decisions that feature accounted for, blank where it was dropped.
`show_structures` renders an electride and a non-electride side by side with ASE (three orientations, unit cell,
element key) — the interstitial space the cavity features measure becomes visible.
`pick_example_structures(swarm, results, cif_dir=...)` selects the most confidently predicted example of each
class automatically.

---

## Configuration (`config.yaml`)

```yaml
swarm:
  n_agents: 5
  n_rounds: 5
  peers_per_agent: 2               # diverse peers shared with each agent per round
  peer_weights: {novelty_to_me: 0.55, novelty_to_selected: 0.30, relevance: 0.15}
  history_rounds: 3                # rounds of private history kept in each agent's prompt
  require_tool_until_round: 3      # rounds where an agent is re-asked if it requests no tool
  proposal_repairs: 1              # same-round retries when a proposal cannot be evaluated
  objective: roc_auc               # what agents maximize; the final table ranks by it
  live_log: false                  # true: print every tool call as it happens instead of round tables

evaluator:
  n_repeats: 5                     # stratified train/validation splits per proposal
  validation_fraction: 0.25
  seed: 42
  max_features: 5                  # features an agent may select per proposal
  min_coverage: 0.5
  gb_params: {n_estimators: 200, learning_rate: 0.05, max_depth: 3, subsample: 0.9}

tools:
  enabled: [web_search, calculator, literature_search, fetch_paper, kb_search]
  literature_search: {max_results: 3, min_interval_s: 3.0}
  kb_search: {top_k: 4, chunk_words: 150, overlap_words: 30, ...}
```

Mis-typed tool names and unknown settings raise an error at startup rather than failing mid-run.

---

## Design notes

**Agents are generalists, not specialists.** No agent is assigned a role. Diversity comes from independent
private histories plus peer exchange, not from a division of labour.

**Peers are chosen for difference.** Each round an agent sees two peers whose answers are, by sentence-embedding
similarity, unlike its own and unlike each other, while still on topic. No agent sees another agent's
current-round output, so rounds are synchronous and no agent can simply chase the leader within a round.

**Evidence precedes the answer.** A tool request is not an answer: the tool is executed, the real result is
handed back to the same agent, and only then is a final answer and proposal produced for that round. Failed
searches are reported as failures ("FAILED … rate_limited"), never as "no literature exists".

**Determinism where it matters.** Same proposal, same splits, same seeds, same numbers. The LLM is the only
stochastic part, and every one of its proposals is checked by the same fixed procedure.

**Honest reporting.** Agents only ever see validation numbers. In-sample metrics of the exported model are
labelled "optimistic" wherever they appear.

---

## Known limitations

* **Selection optimism.** 20–30 proposals are evaluated on the same splits and the best is reported, so the
  winner's score is somewhat optimistic. Treat an external dataset as the real test.
* **Feature coverage.** Agents learn how good a feature is only after using it, so a strong but untried feature
  can go unexplored. Varied starting points (higher temperature) and peer crossover help, but do not guarantee
  coverage. `plot_feature_usage` shows how much of the catalog was ever tried.
* **arXiv rate limits.** All literature calls share one lock with a ≥3 s interval. Repeated runs in quick
  succession still earn HTTP 429 or timeouts; failures are surfaced to the agents and the run continues.
* **Citation discipline.** Agents sometimes paraphrase a paper they have only downloaded, not read. Check the
  run log when a claim matters.
* **Small datasets.** With a few hundred structures, differences under the reported ± spread are noise. The
  feedback says so, but the agents can still over-read them.

---

## Reproducing a run

Everything needed is in the run log: the full config, the data fingerprint, every prompt, every tool result and
every evaluation. Fixing `seed` fixes the splits; the LLM remains stochastic, so re-running produces different
proposals but identical scoring of any given proposal.

---

## Acknowledgements

Built with [instructor](https://github.com/jxnl/instructor) for schema-validated LLM output,
[sentence-transformers](https://www.sbert.net/) for peer selection and retrieval,
[arxiv-mcp-server](https://pypi.org/project/arxiv-mcp-server/) for literature access,
[PyMuPDF](https://pymupdf.readthedocs.io/) for PDF parsing, [scikit-learn](https://scikit-learn.org/) for the
classifier and metrics, and [ASE](https://wiki.fysik.dtu.dk/ase/) for structure visualisation.
