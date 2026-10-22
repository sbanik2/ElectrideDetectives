"""Everything agentic: config loading, schemas, prompts, peer selection, the agent round and the swarm loop.

    from agent import Swarm, load_config
    swarm = Swarm(load_config("config.yaml"), llm=llm, model_name=MODEL_NAME)   # llm: instructor client built in the notebook
    last_round, results = await swarm.run()      # results: leaderboard, selected, final_fit, history
"""
import os, re, json, asyncio, functools, datetime, contextvars
from typing import Literal, Optional

import yaml
import pandas as pd
import networkx as nx
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import tools       
from evaluator import FeatureSetEvaluator


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================ schemas
class SelectedFeature(BaseModel):
    function: str = Field(description="Exact function name from the retrieved RAG feature catalog.")
    feature: str = Field(description="Exact feature name listed under the selected function in the RAG catalog.")
    physical_role: str = Field(description=(
        "What this feature measures physically and why it should separate electrides from non-electrides, "
        "including whether you expect it to be higher or lower in electrides."))


class FeatureSet(BaseModel):
    hypothesis: str = Field(description="Scientific hypothesis for distinguishing electrides from non-electrides.")
    features: list[SelectedFeature] = Field(
        description="The features your model should be built from, copied exactly from the retrieved RAG catalog.",
        min_length=1)
    reasoning: str = Field(description=(
        "Why these features together capture electride character: what each contributes, what you kept from your "
        "previous set and the peers, what you changed and why."))
    suggested_new_features: list[str] = Field(
        default_factory=list,
        description="Additional scientifically relevant features not available in the retrieved catalog, with a brief "
                    "explanation of what they should measure. Do not list unavailable features in `features`.")


class ScientificAgent(BaseModel):
    answer: str = Field(description="Scientific explanation of your electride hypothesis and current model; distinguish evidence from assumptions.")
    next_step: str = Field(description="What you plan to test next and why: a change to your feature set, or a literature question.")
    tool: Literal["none", "web_search", "literature_search", "fetch_paper", "kb_search", "calculator"] = Field(
        description="""literature_search: find arXiv papers (abstracts); fetch_paper: download an arXiv paper's full text into the local collection (input = arXiv ID); kb_search: full-text passages from the local collection (prefix "[arXiv:ID] " to search one paper); 
        web_search: general web; calculator: arithmetic; otherwise none."""
    )
    tool_input: str = Field(default="", description="Short search query (3-6 keywords), an arXiv ID for fetch_paper, '[arXiv:ID] keywords' to search one fetched paper with kb_search, or an arithmetic expression.")
    proposal: Optional[FeatureSet] = Field(default=None, description=(
        "The feature set your electride model should be built from. It is evaluated deterministically on the precomputed "
        "feature CSV after every round. Required in your final answer every round; null only in a message that requests a tool."))

# ============================================================ prompts
# Placeholders {max_features}, {n_dev}, {feature_catalog}, {objective_name} are filled from the data in Swarm().
SYSTEM_PROMPT_TEMPLATE = """You are an autonomous computational materials scientist in a swarm that searches for an interpretable,
physics-informed descriptor set that identifies ELECTRIDES from precomputed structural and chemical features.
Solve the complete problem independently; do not take a fixed specialist role.
You have tools:
web_search(query) for general web evidence, and calculator(expression) for arithmetic;
literature_search(query) for scientific papers from arXiv (titles + full abstracts);
fetch_paper(arxiv_id) to download a promising paper's full text into the local collection; then use kb_search in a later round to read its details;
kb_search(query) for full-text passages (with file + page) from the team's local paper collection;
  write "[arXiv:2103.12345] keywords" to search only one paper you fetched.
A tool result that says FAILED or unavailable means no evidence was obtained, not that no literature exists.
Prefer literature_search when scientific claims need published support; do not answer entirely from memory.
Search queries must be SHORT: 3-6 specific keywords (e.g. "electride interstitial anionic electrons" or
"electride high-throughput screening descriptors"), not sentences.
You may request at most one tool per round. For a tool request, choose the tool and supply a useful, specific tool_input.
After receiving a tool result, use it to revise your model and distinguish sourced evidence from your interpretation.
Cite sources when you use them (arXiv ID, or file + page for kb_search). Assess peers' models and scores critically.
Never invent searches, citations, tool outputs, feature values, metrics, or experimental data.

YOUR PROPOSAL: `proposal` (a FeatureSet), required in your final answer every round.
- features: 1-{max_features} entries, each with function and feature copied EXACTLY from USABLE FEATURES below, plus
  the physical role you expect it to play.
- Python then fits a gradient-boosting classifier on exactly those features and evaluates it. You decide WHICH physical
  quantities the model may use and why; the data decide how to combine them.
- YOUR OBJECTIVE: maximize the validation {objective_name} of that model. ROC-AUC measures how well it RANKS electrides
  above non-electrides, independent of the cutoff; recall and rejection rates also depend on where the cutoff sits.
- A feature set is strong when its features are individually informative AND complementary: two features that measure
  nearly the same quantity add little, and the feedback reports which of your features are redundant.
- The labeled set has {n_dev} structures: the feedback states, split by split, whether a set scores higher or lower
  than your previous best, or within noise. Every feature you add must earn its place physically.
- Descriptors you need but cannot build from the list (e.g. numeric aggregations of JSON features, electron
  localization function maxima) belong in suggested_new_features, never in `features`.
- After every round you receive a deterministic evaluation of your feature set on all {n_dev} labeled structures
  (mean ± spread over repeated train/validation splits), and you see two peers' feature sets with their scores.

HOW EACH ROUND WORKS: your feature set evolves.
- Round 1: choose a first feature set from the literature and the physics below.
- Every later round: propose a NEW feature set starting from your best set so far (marked PARENT in your lineage),
  the peer sets you are shown and the evaluation. Nothing in the parent is fixed. A change can be any of these,
  and each is as valid as the others:
  * replace a feature: when the evaluation shows one is weak (little effect, redundant, or it does not separate the
    classes), swap it for another catalog feature that measures the physics better (e.g. an average instead of a
    minimum, the mean clearance instead of the single largest one);
  * add a feature that captures physics your set is missing, especially for the class that currently fails;
  * remove a feature that adds nothing, so the model relies on fewer, better-understood quantities;
  * borrow a feature from a peer's set when its evaluation shows it carries the score (crossover).
  Several changes can be combined in one round.
- Base every change on the physics, the literature and the evaluation (which class fails, which features carry the
  score, which structures are classified wrong). In `reasoning`, say what you kept, what you changed and why.
- A feature set identical to one already evaluated, yours or a peer's, gives no new information.

ELECTRIDE PHYSICS to weigh (verify with literature_search / kb_search rather than assuming): electrides are ionic
compounds whose excess valence electrons are not bound to atoms but localize in interstitial cavities, channels or
layers and act as anions. Criteria discussed in the literature include an excess of valence electrons relative to a
charge-balanced oxidation-state assignment; strongly electropositive cations (alkali, alkaline-earth, rare-earth; low
electronegativity); empty interstitial space large enough to hold a localized electron and not occupied by anions;
and the connectivity of that space (0D cavities, 1D channels, 2D layers). Categorical labels such as a space-group
number are identifiers, not physical magnitudes.

Keep answer under ~300 words.
Return the ScientificAgent Pydantic fields, including proposal.

{feature_catalog}"""


PROBLEM = """Discover an interpretable, physics-informed descriptor set that identifies electrides.
Electrides are ionic compounds in which some valence electrons are not bound to any atom but occupy
interstitial cavities, channels or layers and act as anions (anionic electrons). Using ONLY the precomputed
structural and chemical features of the labeled structures (catalog in the system prompt), find the set of features
whose combination separates electrides from non-electrides. Justify every selected feature physically, supported by
literature where possible. A model is fitted on your features and evaluated deterministically after every round and the
evaluation is returned to you: use it, with the literature, to choose each new feature set. State what the
available features cannot capture (e.g. where the excess electrons actually localize) and which additional
descriptors would be needed."""

PROPOSAL_REMINDER = (" Also return your complete `proposal` for this round: your next feature set, chosen from your best "
                     "set so far and the peer sets shown, using only usable features and informed by this evidence and "
                     "your evaluation.")

# Which agent an LLM call belongs to (each agent's round runs in its own asyncio task).
CURRENT_AGENT = contextvars.ContextVar("agent", default="main")


# ============================================================ swarm
class Swarm:
    """One configured swarm. Settings come from config.yaml; the LLM client comes from the notebook
    (it differs from system to system). Each run() starts with fresh state."""

    def __init__(self, config, llm, model_name):
        self.cfg = config
        sw = config["swarm"]

        # ---- LLM: an instructor client built in the notebook (endpoint, key, retries, temperature live there)
        self.llm, self.model_name = llm, model_name
        self.token_usage = {}
        self.llm.on("completion:response", self._count_tokens)   # per-agent token accounting

        # ---- one embedding model, shared by peer selection and kb_search
        self.encoder = SentenceTransformer(config["embedding"]["model"])  # cached
        self._embed = functools.lru_cache(maxsize=512)(lambda text: tools._encode([text])[0])

        # ---- precomputed data + deterministic executor; prompt filled from the data
        data, ev = config["data"], config["evaluator"]
        self.objective = sw.get("objective", sw.get("selection_metric", "roc_auc"))   # what agents maximize; final ranking
        self.evaluator = FeatureSetEvaluator(data["feature_catalog"], data["dataset_csv"], **ev, objective=self.objective)
        print(self.evaluator.summary())
        self.problem = PROBLEM
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            max_features=self.evaluator.max_features,
            n_dev=len(self.evaluator.data), feature_catalog=self.evaluator.catalog_prompt(),
            objective_name={"roc_auc": "ROC-AUC", "f1": "F1"}.get(self.objective, self.objective))

        # ---- tools (literature / RAG / web / arithmetic)
        self.tools = tools.setup_tools(config, self.encoder)
        schema_tools = set(ScientificAgent.model_fields["tool"].annotation.__args__) - {"none"}
        missing = set(self.tools) - schema_tools
        if missing:
            raise ValueError(f"Enabled tools {sorted(missing)} are not in ScientificAgent.tool; add them there and in the prompt")
        print(self.problem)
        print("Tools:", list(self.tools))

        # ---- communication graph
        self.n_agents, self.n_rounds = sw["n_agents"], sw["n_rounds"]
        if sw["graph"] != "complete":
            raise ValueError(f"Unsupported swarm.graph {sw['graph']!r}; only 'complete' is implemented")
        self.graph = nx.complete_graph([f"A{i+1}" for i in range(self.n_agents)])
        self.peers_per_agent = sw["peers_per_agent"]
        self.peer_weights = sw["peer_weights"]
        self.history_rounds = sw["history_rounds"]            # each round = 2 messages: (question + tool evidence) -> final answer
        self.peer_evidence_chars = sw["peer_evidence_chars"]  # peers see a trimmed tool_result; full text stays in the run log
        self.tool_result_chars = sw["tool_result_chars"]
        self.require_tool_until_round = sw["require_tool_until_round"]
        self.repairs = sw.get("proposal_repairs", 1)          # same-round retries when a proposal cannot be evaluated
        self.leaderboard_top = sw["leaderboard_top"]
        self.live_log = sw.get("live_log", False)             # print every event as it happens (besides the round tables)
        self.max_listed_sources = config["tools"]["kb_search"].get("max_listed_sources", 30)
        self.outputs_dir = config["paths"]["outputs_dir"]
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.reset()

    # ---------------------------------------------------- per-run state
    def reset(self):
        """Fresh state for a new run: private agent histories, feedback, tokens, run log."""
        # Each agent retains a PRIVATE dialogue. Only selected peers are shared at round boundaries.
        self.histories = {agent: [] for agent in self.graph.nodes}
        self.last_feedback, self.agent_attempts, self.results = {}, {}, {}
        self._notes, self._rows = {}, {}                      # per-agent notes / table rows of the current round
        self.suggestions = []                                 # every suggested_new_features entry: round, agent, text
        self.token_usage.clear()
        self.evaluator.reset()
        self.run_log = f"{self.outputs_dir}/log_run_{datetime.datetime.now():%Y-%m-%d_%H%M%S}.jsonl"

    def _count_tokens(self, response):
        u = getattr(response, "usage", None)
        if u is None:
            return
        t = self.token_usage.setdefault(CURRENT_AGENT.get(), {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0})
        t["calls"] += 1
        t["prompt"] += u.prompt_tokens or 0
        t["completion"] += u.completion_tokens or 0
        t["reasoning"] += getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0

    def log_event(self, **fields):
        """Append one full (untrimmed) record with timestamp to the run log."""
        fields = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **fields}
        with open(self.run_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields, ensure_ascii=False) + "\n")

    def _say(self, msg):
        """Event stream (tool calls, models, repairs) printed only with swarm.live_log: true."""
        if self.live_log:
            print(msg)

    def kb_listing(self):
        """The local paper collection, told to the agents each round so they know kb_search has something to read."""
        sources = tools.kb_sources()
        if not sources:
            return ("Local paper collection (kb_search): empty. fetch_paper adds an arXiv paper's full text to it.")
        shown = sources[:self.max_listed_sources]
        lines = [f"- {src}" + (f": {title[:90]}" if title else "") + f" ({n} passages)" for src, title, n in shown]
        if len(sources) > len(shown):
            lines.append(f"- ... and {len(sources) - len(shown)} more")
        return ("Local paper collection (search it with kb_search; '[source] keywords' searches one paper):\n"
                + "\n".join(lines))

    def plot_graph(self):
        import matplotlib.pyplot as plt
        nx.draw_networkx(self.graph, pos=nx.circular_layout(self.graph), node_color="lightblue", node_size=1200)
        plt.title("Swarm communication graph")
        plt.axis("off")
        plt.show()

    # ---------------------------------------------------- diverse peer selection
    # send only a few diverse peer findings per agent (instead of broadcasting everything),
    # using sentence embeddings + cosine similarity from a small open-source local encoder.
    def embed(self, text):
        # Normalized vectors make their dot product equal to cosine similarity.
        return self._embed(text)

    def similarity(self, text_a, text_b):
        if not text_a.strip() or not text_b.strip():
            return 0.0
        return float(self.embed(text_a) @ self.embed(text_b))

    def select_diverse_peers(self, agent, previous_round, k=None):
        k = self.peers_per_agent if k is None else k
        w = self.peer_weights
        candidates = {n: previous_round[n] for n in self.graph.neighbors(agent) if n in previous_round}
        if not candidates:
            return {}
        own_answer = previous_round.get(agent, {}).get("answer", "")
        chosen = {}
        while candidates and len(chosen) < k:
            def score(name):
                answer = candidates[name]["answer"]
                novelty_to_me = 1 - self.similarity(own_answer, answer)
                novelty_to_selected = (1 - max(self.similarity(answer, p["answer"]) for p in chosen.values())) if chosen else 1.0
                topic_relevance = self.similarity(self.problem, answer)
                return (w["novelty_to_me"] * novelty_to_me + w["novelty_to_selected"] * novelty_to_selected
                        + w["relevance"] * topic_relevance)
            best = max(candidates, key=score)
            chosen[best] = candidates.pop(best)
        return chosen

    # ---------------------------------------------------- discovery loop: proposal -> evaluation -> feedback
    def lineage_context(self, agent, round_number=None):
        """Appended to the agent's round question: its feature-set lineage (parent marked), last evaluation, the task."""
        obj, name = self.objective, {"roc_auc": "ROC-AUC", "f1": "F1"}.get(self.objective, self.objective)
        own = [e for e in self.evaluator.history if e["agent"] == agent]
        ok = [e for e in own if e["status"] == "ok"]
        parent = max(ok, key=lambda e: (e["metrics_mean"][obj], e["id"])) if ok else None
        lines = []
        for e in own:
            if e["status"] == "ok":
                m, sd = e["metrics_mean"], e["metrics_std"]
                mark = "   <- PARENT (your best so far)" if parent and e["id"] == parent["id"] else ""
                lines.append(f"- R{e['round']} #{e['id']}: {', '.join(e['features'])}  ROC-AUC {m['roc_auc']:.3f}±{sd['roc_auc']:.3f}, "
                             f"F1 {m['f1']:.2f}, electrides found / non-electrides rejected {m['recall']:.2f} / {m['specificity']:.2f}{mark}")
            else:
                lines.append(f"- R{e['round']} #{e['id']}: {', '.join(e['features'])}  INVALID: {str(e.get('error', ''))[:120]}")
        if round_number:
            missing = sorted(set(range(1, round_number)) - {e["round"] for e in own})
            if missing:
                lines.append("- no evaluated feature set in round(s) " + ", ".join(map(str, missing)))
        feedback = self.last_feedback.get(agent)
        if not lines:
            return ("YOUR FEATURE-SET LINEAGE: none yet.\n\n"
                    "THIS ROUND: choose your first feature set (see HOW EACH ROUND WORKS in the system prompt) and return "
                    "it in `proposal`.")
        return (f"YOUR FEATURE-SET LINEAGE (all your sets so far; validation results; objective = {name}):\n" + "\n".join(lines)
                + (f"\n\nEVALUATION OF YOUR LAST FEATURE SET:\n{feedback}" if feedback else "")
                + "\n\nTHIS ROUND: propose your next feature set, starting from your PARENT, the peer sets above and the "
                  "evaluation (see HOW EACH ROUND WORKS in the system prompt), and return it in `proposal`.")

    async def evaluation_step(self, agent, round_number, convo, response, draft=None):
        """End of round: make sure there is a proposal, execute it deterministically, allow a repair."""
        tag = f"[{agent} | Round {round_number}]"
        convo = convo + [{"role": "assistant", "content": response.model_dump_json()}]
        proposal = response.proposal or draft
        if proposal is None:
            self._say(f"{tag} No proposal returned; asking for one")
            self._notes.setdefault(agent, []).append("no proposal returned; asked again")
            response = await self.llm.chat.completions.create(
                model=self.model_name, response_model=ScientificAgent, messages=convo + [{"role": "user", "content":
                    "You did not include `proposal`. Return your answer again with a complete FeatureSet in `proposal` "
                    "(usable features only). Set tool='none' and tool_input=''."}])
            convo.append({"role": "assistant", "content": response.model_dump_json()})
            proposal = response.proposal
        result = None
        for attempt in range(self.repairs + 1):
            if proposal is None:
                result = {"status": "invalid", "summary": "INVALID (no proposal returned)",
                          "feedback": "You returned no proposal; one is required every round."}
                break
            result = await asyncio.to_thread(self.evaluator.evaluate, proposal, agent, round_number)
            self._say(f"{tag} MODEL: {result['summary']}")
            if result["status"] == "ok" or attempt == self.repairs:
                break
            self._say(f"{tag} Proposal could not be evaluated; asking agent to correct it")
            self._notes.setdefault(agent, []).append(f"repaired: {result['error'][:120]}")
            response = await self.llm.chat.completions.create(
                model=self.model_name, response_model=ScientificAgent, messages=convo + [{"role": "user", "content":
                    f"Your proposal could not be evaluated:\n{result['error']}\n\nReturn a corrected `proposal` (keep the "
                    "scientific idea if it is sound; use only usable features). Set tool='none' and tool_input=''."}])
            convo.append({"role": "assistant", "content": response.model_dump_json()})
            proposal = response.proposal or proposal
        if proposal is not None and response.proposal is None:
            response = response.model_copy(update={"proposal": proposal})   # history shows what was evaluated
        self.last_feedback[agent] = f"(from round {round_number}) {result['feedback']}"
        self.agent_attempts.setdefault(agent, []).append(f"R{round_number}: {result['summary']}")
        return response, result

    def finalize(self):
        """Select the best model deterministically, refit it on all labeled structures and export it."""
        ev = self.evaluator
        board = ev.leaderboard(top=self.leaderboard_top)       # ranked by the objective's validation mean
        best = ev.select_best()
        model_path = self.run_log.replace("log_run_", "final_model_").replace(".jsonl", ".joblib")
        final_fit = ev.final_fit(best, model_path=model_path) if best else {"status": "no feature set was evaluated"}
        _show(pd.DataFrame([{"Rank": i + 1, "": "★" if best and r["id"] == best["id"] else "", "#": r["id"], "Agent": r["agent"],
                             "Round": r["round"], "Features": r["n_features"],
                             "F1": f"{r['metrics_mean']['f1']:.2f} ± {r['metrics_std']['f1']:.2f}",
                             "ROC-AUC": f"{r['metrics_mean']['roc_auc']:.2f} ± {r['metrics_std']['roc_auc']:.2f}",
                             "Electrides found / non-electrides rejected":
                                 f"{r['metrics_mean']['recall']:.2f} / {r['metrics_mean']['specificity']:.2f}",
                             "Feature set (share of the model's decisions)": _feature_shares(r)} for i, r in enumerate(board)]),
              f"Best feature sets ranked by validation {'ROC-AUC' if self.objective == 'roc_auc' else self.objective}: top "
              f"{len(board)} of {len(ev.leaderboard())} unique sets ({len(ev.history)} evaluations; mean ± sd over repeated "
              "splits; ★ = top set, refit and exported)")
        if best:
            path = self.run_log.replace("log_run_", "final_model_").replace(".jsonl", ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"selected": best, "final_fit": final_fit, "leaderboard": board,
                           "suggested_features": _group_suggestions(self.suggestions)}, f, indent=2, default=str)
            m = final_fit["in_sample_metrics"]
            _show(pd.DataFrame([
                {"Item": "Model", "Value": f"#{best['id']} ({best['agent']}, round {best['round']})"},
                {"Item": "Ranked first by", "Value": best["selection_rule"]},
                {"Item": f"Fitted on all {final_fit['n_structures']} structures",
                 "Value": f"gradient boosting on {len(final_fit['features'])} features "
                          f"({', '.join(f'{k}={v}' for k, v in final_fit['gb_params'].items())})"},
                *({"Item": col.split("__", 1)[0] + "." + col.split("__", 1)[1],
                   "Value": f"share of the model's decisions {share:.0%}"}
                  for col, share in sorted(final_fit["importances"].items(), key=lambda kv: -kv[1])),
                {"Item": "Validation (repeated splits)", "Value": f"F1 {best['metrics_mean']['f1']:.3f} ± {best['metrics_std']['f1']:.3f}, "
                                                                   f"ROC-AUC {best['metrics_mean']['roc_auc']:.3f} ± {best['metrics_std']['roc_auc']:.3f}"},
                {"Item": "In-sample (optimistic)", "Value": ", ".join(f"{k} {m[k]:.3f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc"))},
                {"Item": "Saved for external testing", "Value": path},
            ]), "Top feature set (model refit on all structures and exported)")
        suggested = _group_suggestions(self.suggestions)
        if suggested:
            _show(pd.DataFrame(suggested[:30]).rename(columns={"feature": "Suggested new feature", "times": "Times suggested",
                                                              "by": "By (agent, round)", "description": "Description (first seen)"}),
                  f"Suggested new features from all agents and rounds ({len(suggested)} distinct)")
        self.log_event(event="final_ranking", leaderboard=board, selected=best, final_fit=final_fit, suggested_features=suggested,
                       n_evaluations=len(ev.history), evaluations=[ev._public(e) for e in ev.history])
        self.results.update(leaderboard=board, selected=best, final_fit=final_fit, suggested_features=suggested,
                            history=[ev._public(e) for e in ev.history])
        return self.results

    # ---------------------------------------------------- one agent, one round
    async def agent_round(self, agent, round_number, previous_round):
        CURRENT_AGENT.set(agent)
        self._notes[agent] = []
        start = dict(self.token_usage.get(agent, {}))
        peers = self.select_diverse_peers(agent, previous_round)
        if peers:
            self._say(f"[{agent} | Round {round_number}] Peer context: {list(peers)}")

        peers_for_prompt = {n: {**p, "tool_result": str(p["tool_result"])[:self.peer_evidence_chars]} for n, p in peers.items()}
        question = (
            f"Scientific problem:\n{self.problem}\n\nRound {round_number}/{self.n_rounds}. "
            f"Peers' models and findings from the PREVIOUS round (critique, don't blindly copy):\n"
            f"{json.dumps(peers_for_prompt, indent=2) if peers else 'None; solve independently.'}\n\n"
            "Select a tool if you need external evidence or arithmetic. "
            "In Round 1, first investigate the published scientific evidence using literature_search (or web_search); "
            "do not present a fully researched answer before checking a source."
        )
        question += "\n\n" + self.kb_listing()
        question += "\n\n" + self.lineage_context(agent, round_number)
        history = self.histories[agent]
        messages = [
            {"role": "system", "content": self.system_prompt},
            *(history[-2 * self.history_rounds:] if self.history_rounds else history),
            {"role": "user", "content": question}
        ]
        response = await self.llm.chat.completions.create(
            model=self.model_name, response_model=ScientificAgent, messages=messages
        )

        # Give the LLM another chance to request evidence in the first few rounds.
        # We do NOT silently invent or force a tool call on its behalf.
        if round_number <= self.require_tool_until_round and response.tool == "none":
            self._say(f"[{agent} | Round {round_number}] No tool requested; asking agent to gather evidence first")
            self._notes[agent].append("chose no tool; asked to gather evidence")
            response = await self.llm.chat.completions.create(
                model=self.model_name, response_model=ScientificAgent,
                messages=messages + [{
                    "role": "user",
                    "content": "Before finalizing, request ONE tool that would most strengthen your evidence: "
                               "literature_search (short 3-6 keyword query), fetch_paper (an arXiv ID from earlier results), "
                               "or kb_search (short query on downloaded/local papers). Set tool and tool_input accordingly."
                }]
            )

        requested_tool = response.tool
        requested_input = response.tool_input
        draft_proposal = response.proposal   # a model drafted together with the tool request, if any
        round_messages = messages
        tool_result = "Not requested"
        if response.tool in self.tools and response.tool_input.strip():
            self._say(f"[{agent} | Round {round_number}] TOOL CALL: {response.tool}")
            self._say(f"[{agent} | Round {round_number}] INPUT: {response.tool_input}")
            try:
                tool_result = await asyncio.to_thread(self.tools[response.tool], response.tool_input)
            except Exception as exc:
                tool_result = f"Tool error: {type(exc).__name__}: {exc}"
            self._say(f"[{agent} | Round {round_number}] RESULT: {str(tool_result)[:500]}")

            # A tool selection is NOT yet a grounded answer. Let the same agent
            # interpret the actual returned evidence during this round.
            round_messages = messages + [
                    {"role": "assistant", "content": response.model_dump_json()},
                    {"role": "user", "content": (
                        f"Executed {response.tool}({response.tool_input!r}). "
                        f"Actual tool result:\n{str(tool_result)[:self.tool_result_chars]}\n\n"
                        "Now give your revised scientific answer grounded in this result. "
                        "Do not request another tool in this round: set tool='none' and tool_input=''. "
                        "If the search failed or returned no usable evidence, say so clearly."
                        + PROPOSAL_REMINDER
                    )}
                ]
            response = await self.llm.chat.completions.create(
                model=self.model_name, response_model=ScientificAgent, messages=round_messages)
        elif response.tool == "none":
            self._say(f"[{agent} | Round {round_number}] No tool called")
        else:
            self._say(f"[{agent} | Round {round_number}] Tool not executed: "
                      f"tool={response.tool!r}, input={response.tool_input!r}")
            self._notes[agent].append(f"requested {response.tool!r} but it was not executed (disabled or empty input)")

        # Deterministic evaluation of the agent's feature set; its feedback opens the agent's next round.
        response, evaluation = await self.evaluation_step(agent, round_number, round_messages, response, draft_proposal)

        record = {
            "answer": response.answer,
            "next_step": response.next_step,
            "tool": requested_tool,
            "tool_input": requested_input,
            "tool_result": tool_result,
        }
        record["proposal"] = response.proposal.model_dump() if response.proposal else None
        for text in (response.proposal.suggested_new_features if response.proposal else []):
            if str(text).strip():
                self.suggestions.append({"round": round_number, "agent": agent, "suggestion": str(text).strip()})
        record["evaluation"] = evaluation["summary"]
        tokens = {k: v - start.get(k, 0) for k, v in self.token_usage.get(agent, {}).items()}
        self._say(f"[{agent} | Round {round_number}] TOKENS: {tokens}")
        if evaluation.get("duplicate_of"):
            self._notes[agent].append(f"same model as #{evaluation['duplicate_of']}")
        progress = evaluation.get("progress") or {}
        if progress.get("verdict") in ("improved", "worse", "no clear change"):
            self._notes[agent].append(f"{progress['verdict']} vs parent #{progress['reference_id']} ({progress['delta']:+.3f})")
        self._rows[agent] = _round_row(agent, peers, requested_tool, requested_input, tool_result,
                                       response, evaluation, self._notes[agent], tokens)
        if not self.live_log:
            print(agent, end=" ", flush=True)                  # progress while the round runs
        self.log_event(event="agent_round", round=round_number, agent=agent, peers=list(peers), question=question, **record, tokens=tokens)

        # Preserve executed tool metadata (even though the post-tool answer sets tool='none').
        # 'tool' stays readable to peers; evidence is in tool_result.
        # Evidence BEFORE the answer that used it, and user/assistant strictly alternating.
        evidence = (f"\n\n[Executed {requested_tool}({requested_input!r}) this round]\nResult: {str(tool_result)[:self.tool_result_chars]}"
                    if tool_result != "Not requested" else "\n\n[No tool executed this round]")
        history.extend([
            {"role": "user", "content": question + evidence},
            {"role": "assistant", "content": response.model_dump_json()},
        ])
        return agent, record

    # ---------------------------------------------------- the swarm
    async def run(self):
        """N rounds of all agents in parallel, then deterministic selection + refit + export.
        Returns (last round's records by agent, results dict with leaderboard / selected / final_fit / history)."""
        self.reset()
        print("Run log:", self.run_log)
        self.log_event(event="config", model=self.model_name, n_agents=self.n_agents, n_rounds=self.n_rounds,
                       peers_per_agent=self.peers_per_agent, tools=list(self.tools), problem=self.problem,
                       evaluation=self.evaluator.config(), settings=self.cfg)
        previous = {}
        for r in range(1, self.n_rounds + 1):
            print(f"\n{'='*20} ROUND {r}/{self.n_rounds} {'='*20}")
            if not self.live_log:
                print("finished:", end=" ", flush=True)
            # A full synchronous round: no agent can see another agent's current-round output.
            self._rows = {}
            completed = await asyncio.gather(*(self.agent_round(a, r, previous) for a in self.graph.nodes))
            previous = dict(completed)
            print()
            rows = [self._rows[a] for a in self.graph.nodes]
            _show(pd.DataFrame([row["table"] for row in rows]), f"Round {r}: tools, evidence and models")
            _show(pd.DataFrame([row["reasoning"] for row in rows]), f"Round {r}: reasoning")
            _plot_round_roc(r, [row["roc"] for row in rows if row["roc"]])

        # Deterministic model selection, refit on all structures + export.
        results = self.finalize()
        total = {k: sum(t.get(k, 0) for t in self.token_usage.values()) for k in ("calls", "prompt", "completion", "reasoning")}
        _show(pd.DataFrame([{"Agent": a, **t} for a, t in sorted(self.token_usage.items())] + [{"Agent": "TOTAL", **total}]),
              "Token usage")
        self.log_event(event="tokens", per_agent=self.token_usage, total=total)
        return previous, results


# ============================================================ display helpers
def _show(df, title):
    """Styled table in Jupyter; plain text elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is None:
            raise RuntimeError("not in IPython")
        display(df.style.hide(axis="index").set_caption(title)
                .set_properties(**{"text-align": "left", "vertical-align": "top", "white-space": "pre-wrap", "font-size": "12px"})
                .set_table_styles([{"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: bold; font-size: 14px; padding: 8px 0;"},
                                   {"selector": "th", "props": "text-align: left; background: #f0f0f0; color: #222;"},
                                   {"selector": "td", "props": "border-top: 1px solid #ddd; padding: 4px 8px;"}]))
    except Exception:
        with pd.option_context("display.max_colwidth", 80, "display.width", 250):
            print(f"\n{title}\n{df.to_string(index=False)}")

def _feature_shares(entry):
    """Each feature with the share of the fitted model's decisions it accounts for, strongest first."""
    fa = entry.get("feature_analysis") or {}
    rows = sorted(fa.values(), key=lambda v: -(v.get("model_importance") or 0))
    return "\n".join(f"{v['feature']}  {(v.get('model_importance') or 0):.0%}" for v in rows) or "-"

def _evidence(tool, result):
    """One line per paper / passage instead of the raw JSON."""
    text = str(result)
    if text == "Not requested":
        return "-"
    start = text.find("[{")
    try:
        data = json.loads(text[start:]) if start >= 0 else None
    except ValueError:
        data = None
    if isinstance(data, list) and data:
        note = text[:start].strip()
        if tool == "literature_search":
            lines = [f"{p.get('arxiv_id')} ({p.get('year')}): {str(p.get('title', ''))[:70]}" for p in data]
        elif tool == "kb_search":
            lines = [f"{c.get('source')}" + (f" p.{c['page']}" if c.get("page") else "") + f" (score {c.get('score')})" for c in data]
        elif tool == "web_search":
            lines = [str(h.get("title", ""))[:80] for h in data]
        else:
            lines = [text[:160]]
        return "\n".join(([note] if note else []) + lines)
    return text[:200]

def _round_row(agent, peers, tool, tool_input, tool_result, response, evaluation, notes, tokens):
    p = response.proposal
    roc = None
    if evaluation.get("status") == "ok":
        mm, ms = evaluation["metrics_mean"], evaluation["metrics_std"]
        f1, auc = f"{mm['f1']:.2f} ± {ms['f1']:.2f}", f"{mm['roc_auc']:.2f} ± {ms['roc_auc']:.2f}"
        by_class = f"{mm['recall']:.2f} / {mm['specificity']:.2f}"
        roc = {"agent": agent, "features": _feature_shares(evaluation),
               "auk": 0, "auc": mm["roc_auc"], "auc_sd": ms["roc_auc"], "f1": mm["f1"], "f1_sd": ms["f1"],
               **evaluation.get("roc", {"labels": [], "probabilities": []})}
    else:
        f1, auc, by_class = "invalid", "-", "-"
        notes = notes + [evaluation.get("summary", "")]
    return {"table": {
                "Agent": agent,
                "Peers": ", ".join(peers) or "-",
                "Tool": "-" if tool == "none" else f"{tool}\n\"{tool_input}\"",
                "Evidence": _evidence(tool, tool_result),
                "Feature set": "\n".join(f.feature for f in p.features) if p else "-",
                "Share of the model's decisions": _feature_shares(evaluation) if evaluation.get("status") == "ok" else "-",
                "F1": f1, "ROC-AUC": auc, "Electrides found / non-electrides rejected": by_class,
                "Notes": "\n".join(notes) or "-",
                "Tokens in / out": f"{tokens.get('prompt', 0):,} / {tokens.get('completion', 0):,}"},
            "roc": roc,
            "reasoning": {
                "Agent": agent,
                "Answer": (response.answer[:450] + " ...") if len(response.answer) > 450 else response.answer,
                "Next step": response.next_step[:250],
                "Suggested new features": "\n".join(f"• {str(t)[:160]}" for t in p.suggested_new_features) if p and p.suggested_new_features else "-"}}


def _plot_round_roc(round_number, rocs):
    """Validation ROC curve of every feature set evaluated in this round (predictions pooled over the splits);
    the round's best feature set (highest ROC-AUC, then F1) is drawn bold, then summarized."""
    rocs = [r for r in rocs if len(set(r["labels"])) == 2]
    if not rocs:
        print(f"Round {round_number}: no executable model to plot.")
        return
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    best = max(rocs, key=lambda r: (round(r["auc"], 6), round(r["f1"], 6)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in sorted(rocs, key=lambda r: r is best):          # best drawn last, on top
        fpr, tpr, _ = roc_curve(r["labels"], r["probabilities"])
        is_best = r is best
        ax.plot(fpr, tpr, lw=3.2 if is_best else 1.3, alpha=1.0 if is_best else 0.65, drawstyle="steps-post",
                label=f"{r['agent']}{'  ★ best' if is_best else ''}   AUC {r['auc']:.2f} ± {r['auc_sd']:.2f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Non-electrides wrongly called electrides (false-positive rate)")
    ax.set_ylabel("Electrides detected (recall)")
    ax.set_title(f"Round {round_number}: validation ROC per model (pooled over splits)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"Best feature set in round {round_number}: {best['agent']} | ROC-AUC {best['auc']:.2f} ± {best['auc_sd']:.2f} | "
          f"F1 {best['f1']:.2f} ± {best['f1_sd']:.2f} | {best['features'].replace(chr(10), ', ')}")


def _group_suggestions(items):
    """Group free-text feature suggestions by their name (text before ':', ' - ', ' – ' or '('), most frequent first."""
    groups = {}
    for it in items:
        text = it["suggestion"]
        name = re.split(r"\s*(?::|\s[-–—]\s|\()", text, maxsplit=1)[0].strip().strip("`*'\"") or text[:60]
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        g = groups.setdefault(key, {"feature": name[:80], "times": 0, "by": [], "description": text[:220]})
        g["times"] += 1
        g["by"].append(f"{it['agent']} R{it['round']}")
    rows = sorted(groups.values(), key=lambda g: (-g["times"], g["feature"].lower()))
    for g in rows:
        g["by"] = ", ".join(dict.fromkeys(g["by"]))
    return rows
