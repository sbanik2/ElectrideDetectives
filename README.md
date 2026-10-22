<p align="center">
  <img src="docs/logo.png" alt="ElectrideDetectives" width="160">
</p>

<h1 align="center">Agentic Swarms for Electride Materials Discovery</h1>

<p align="center">
  <a href="#how-a-round-works">How it works</a> ·
  <a href="#the-agents-learn-and-share">Learning and sharing</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#running">Running</a> ·
  <a href="#analysis-plots-srcanalysispy">Analysis</a> ·
  <a href="#configuration-configyaml">Configuration</a>
</p>

A multi-agent swarm in which several LLM scientists independently reason about **why** a material is an
electride, propose which physical descriptors a classifier should be built from, and refine those proposals
over several rounds with evidence from the data.

Electrides are ionic compounds in which some valence electrons are not bound to any atom: they occupy
interstitial cavities, channels or layers and behave as anions. Deciding which measurable structural and
chemical quantities reveal that behaviour is a scientific judgement, not a fitting problem. This project puts
the judgement in the hands of reasoning agents and leaves the arithmetic to a fixed, auditable backend.

---

## How a round works

<p align="center">
  <img src="docs/diagram.png"
       alt="One round: agents gather evidence with tools, propose a feature set, receive a per-class evaluation, then exchange findings with peers"
       width="100%">
</p>

Each round, every agent may call one tool for real evidence, then returns a **FeatureSet**: the descriptors its
model should be built from, each with the physical role it is expected to play, plus the hypothesis and the
reasoning behind the choice. The backend fits a classifier on exactly those features and returns a per-class
evaluation, which opens the agent's next round.

<details>
<summary>Text version of the diagram</summary>

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
  │ evaluation                                                                 │             │
  │  • classifier fitted on exactly those features                             │             │
  │  • 5 stratified train/validation splits of ALL labeled structures          │             │
  │  • per-class metrics, per-feature role, hardest structures, parent compare │             │
  └────────┬───────────────────────────────────────────────────────────────────┘             │
           │ feedback (plain words + numbers)                                                 │
           ▼                                                                                 │
  Round r+1: each agent sees its own lineage + feedback, and 2 diverse peers' proposals ──────┘
```

</details>

After the last round the top-ranked feature set is refit on all labeled structures and exported for scoring
new data.

---

## The agents learn and share

Between rounds, knowledge moves in two directions.

**Each agent learns from its own history.** An agent keeps a private lineage: every feature set it has
proposed, the score each one reached, and which one is currently its best — its *parent*. The next proposal is
written from that parent, so an agent builds on what its own evidence showed instead of starting over.

**Agents share what they found with each other.** At every round boundary each agent is shown two peers'
proposals, with their scores and their reasoning. Peers are picked for difference rather than quality: an agent
sees colleagues whose findings are unlike its own and unlike each other, so useful descriptors spread through
the swarm while the agents keep exploring different parts of the feature space. An agent may adopt a peer's
descriptor when the evidence supports it, and the prompt asks it to state what it kept, what it changed, and why.

**What an agent reads before deciding.** The evaluation of its last proposal arrives in plain language first,
then in numbers:

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
├── run_swarm.ipynb                               # the only notebook: LLM setup → config → build → run → analysis
├── config.yaml                                   # every structural setting (no secrets, no LLM endpoint)
├── requirements.txt
├── src/
│   ├── agent.py                                  # schemas, prompts, peer selection, agent round, swarm loop
│   ├── evaluator.py                              # deterministic evaluation, ranking, export, score_dataset
│   ├── tools.py                                  # web_search, calculator, arXiv search/download, local-paper RAG
│   └── analysis.py                               # post-run plots (progress, feature maps, metrics, structures)
├── data/
│   ├── electride_feature_rag_clean.jsonl         # feature catalog: function, feature, description
│   ├── electride_features_dataset_200_sample.csv # precomputed features + labels (200 structures)
│   └── data_extract/                             # how those features were computed: notebook, script,
│                                                 #   labels.json, example CIFs
├── docs/                                         # logo and diagram
├── inputs/kb/                                    # local PDFs + papers the agents download
└── outputs/                                      # run logs, exported models, scores
```

### Responsibilities

* **`src/agent.py`** — the `FeatureSet` / `ScientificAgent` schemas, the system prompt, the problem statement,
  the communication graph, diverse peer selection, the per-round loop, the final ranking and export, and the
  tables/plots printed during a run. Receives a ready LLM client from the notebook.
* **`src/evaluator.py`** — `FeatureSetEvaluator` (what the swarm calls), `evaluate_feature_set` (one fit + one
  split), and `score_dataset` (apply an exported model to any CSV). No LLM calls anywhere in this file.
* **`src/tools.py`** — one function per tool, each taking a single string and always returning a string, so an
  agent always sees either evidence or an explicit failure message.
* **`src/analysis.py`** — read-only plots over `swarm.evaluator.history`; nothing is refitted.
* **`data/data_extract/`** — the feature-generation step that produced the CSV, kept separate from the swarm.
  See its own [README](data/data_extract/README.md).

---

## Input data

### 1. Feature catalog — `data/electride_feature_rag_clean.jsonl`

One JSON record per line:

```json
{"function": "cavity_geometry", "feature": "max_clearance_A", "description": "Distance from the void centre to the nearest atom's covalent surface, in Å."}
```

The catalog is what the agents see. Descriptions matter: they are the only thing telling an agent what a
feature physically means.

### 2. Feature table — `data/electride_features_dataset_200_sample.csv`

The sample shipped here holds **200 labeled structures: 84 electrides and 116 non-electrides**, with one row
per structure:

| column | meaning |
|---|---|
| `structure` | identifier (e.g. `mp-11854.cif`) |
| `label` | `1` = electride, `0` = non-electride |
| `<function>__<feature>` | one column per catalog entry, e.g. `cavity_geometry__max_clearance_A` |

`true`/`false` strings are converted to 1/0; missing values are imputed with the **training fold's** median.

A catalog feature is offered to the agents only if its column exists, is numeric, varies, and is present in at
least `min_coverage` of rows — 26 of the 49 catalog entries qualify in the shipped sample. JSON-valued columns
(lists, nested records) and constant run settings are listed to the agents as *not usable*, so they can request
numeric aggregations of them instead.

To build this table for your own structures, use `data/data_extract/` (`electride_features.py` and
`Electride_Feature_CSV.ipynb`, with labels in `labels.json` and CIFs in `data/data_extract/structures/`).

### 3. Local papers — `inputs/kb/`

Drop PDFs in here. They are chunked, embedded with `all-MiniLM-L6-v2` and searchable with `kb_search`.
Papers that agents download with `fetch_paper` land in the same folder and persist across runs. Each round the
agents are told what the collection currently contains, so they can target one paper:
`kb_search("[Assessing the design rules of electrides.pdf] mean clearance")`.
Scanned PDFs with no text layer are reported at index time ("no extractable text … run OCR first").

---

## Installation

Python **3.11+** (required by the arXiv MCP server).

```bash
git clone https://github.com/sbanik2/ElectrideDetectives.git
cd ElectrideDetectives
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

Run `run_swarm.ipynb` from the repository root. The modules live in `src/`, so the notebook puts that folder on
the import path first:

```python
import sys; sys.path.insert(0, "src")
from agent import Swarm, load_config
from evaluator import score_dataset

CONFIG = load_config("config.yaml")
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
# in a later session, newest exported model:
scores, metrics = score_dataset(None, "my_new_structures.csv")
```

The new CSV needs only the feature columns the model uses. `label` is optional; when present you also get
metrics and a confusion matrix. The same training medians are reused for missing values.

---

## Analysis plots (`src/analysis.py`)

```python
import analysis
progress = analysis.plot_agent_progress(swarm)   # score per agent per round + swarm best-so-far
analysis.plot_feature_map(swarm)                 # per agent: which features, which round, how much used
analysis.plot_feature_usage(swarm)               # how many agents picked each feature, per round
metrics = analysis.plot_final_model(swarm, results)  # ROC, PR, confusion matrix, score separation
analysis.show_structures("data/data_extract/structures/mp-11854.cif",
                         "data/data_extract/structures/mp-559931.cif", repeat=(2, 2, 1))
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

data:
  feature_catalog: data/electride_feature_rag_clean.jsonl
  dataset_csv: data/electride_features_dataset_200_sample.csv

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

---

## Reproducing a run

Everything needed is in the run log: the full config, the data fingerprint, every prompt, every tool result and
every evaluation. Fixing `seed` fixes the splits; the LLM remains stochastic, so re-running produces different
proposals but identical scoring of any given proposal.

---

## License

MIT — see [LICENSE](LICENSE).
