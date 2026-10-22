"""Deterministic evaluation of an agent's feature set with gradient boosting (no LLM calls in this file).

evaluate_feature_set   fit a gradient-boosting classifier on the agent's selected features, score one split
FeatureSetEvaluator    what the swarm calls: scores every proposal on ALL labeled structures over repeated
                       stratified train/validation splits, detects repeated proposals, keeps the ranking,
                       writes the agents' feedback, refits and exports the top model
score_dataset          apply an exported model to any feature CSV

Proposals are the agent's FeatureSet objects (agent.py); this file only reads their
.features (function, feature) and .hypothesis.
"""
import difflib, glob, json, math, os, threading
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

METRICS = ("f1", "roc_auc", "accuracy", "precision", "recall", "specificity")
METRIC_NAMES = {"f1": "F1", "roc_auc": "ROC-AUC", "accuracy": "accuracy", "precision": "precision",
                "recall": "electride recall", "specificity": "non-electride rejection (specificity)"}
GB_DEFAULTS = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9}


# ============================================================ catalog
FEATURE_CAPABILITIES, ALLOWED_PAIRS = [], set()

def load_catalog(path="electride_feature_rag_clean.jsonl"):
    """Read the RAG feature catalog (one JSON record per line) and set the module defaults."""
    global FEATURE_CAPABILITIES, ALLOWED_PAIRS
    with open(path, encoding="utf-8") as f:
        FEATURE_CAPABILITIES = [json.loads(line) for line in f if line.strip()]
    ALLOWED_PAIRS = {(r["function"], r["feature"]) for r in FEATURE_CAPABILITIES}
    return FEATURE_CAPABILITIES

def to_numeric(series):
    """'true'/'false' -> 1/0, everything else numeric or NaN."""
    values = series.map(lambda v: 1 if str(v).strip().lower() == "true"
                        else 0 if str(v).strip().lower() == "false" else v)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


# ============================================================ one evaluation
def _balanced_weight(y):
    return np.where(y == 1, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((y == 0).sum(), 1)))

def evaluate_feature_set(proposal, csv_path="electride_features_dataset.csv", seed=42,
                         validation_fraction=0.25, n_permutations=10, *, data=None, split=None,
                         allowed_pairs=None, gb_params=None):
    """
    Fit a gradient-boosting classifier on the selected features and evaluate electride classification.

    Returns numerical metrics, feature-level statistics (overall and per class), correlations and the
    validation predictions. `data` (DataFrame) replaces csv_path; `split` gives explicit
    (train_idx, valid_idx) row positions in that frame instead of a random stratified split.
    """
    # 1. LOAD DATA
    df = data.copy() if data is not None else pd.read_csv(csv_path)
    if not {"structure", "label"}.issubset(df.columns):
        raise ValueError("CSV requires 'structure' and 'label' columns")
    if split is None:
        df = df.dropna(subset=["label"]).reset_index(drop=True)
    elif df["label"].isna().any():
        raise ValueError("With an explicit split, data must have no missing labels")
    if (len(df) < 8 or df["label"].nunique() != 2 or df["label"].value_counts().min() < 3
            or not set(df["label"].unique()).issubset({0, 1})):
        raise ValueError("Need at least 8 structures, with at least 3 per class")

    # 2. RETRIEVE THE SELECTED FEATURES FROM THE CSV
    pairs = ALLOWED_PAIRS if allowed_pairs is None else allowed_pairs
    X = pd.DataFrame(index=df.index)
    feature_definitions = {}
    for f in proposal.features:
        pair = (f.function, f.feature)
        if pair not in pairs:
            raise ValueError(f"Feature not in catalog: {pair}")
        column = f"{f.function}__{f.feature}"
        if column not in df.columns:
            raise ValueError(f"Missing CSV column: {column}")
        numeric = to_numeric(df[column])
        if numeric.notna().sum() == 0:
            raise ValueError(f"{column} has no scalar numerical values")
        X[column] = numeric
        feature_definitions[column] = {"function": f.function, "feature": f.feature, "csv_column": column}

    y = df["label"].astype(int).to_numpy()
    columns = list(X.columns)

    # 3. SPLIT (stratified: both classes in both halves) AND IMPUTE
    if split is None:
        train_idx, valid_idx = train_test_split(np.arange(len(df)), test_size=validation_fraction,
                                                random_state=seed, stratify=y)
    else:
        train_idx, valid_idx = (np.asarray(s, dtype=int) for s in split)
    y_train, y_valid = y[train_idx], y[valid_idx]
    if len(np.unique(y_valid)) < 2 or len(np.unique(y_train)) < 2:
        raise ValueError("Training and validation splits must both contain electrides and non-electrides")

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X.iloc[train_idx])
    X_valid = imputer.transform(X.iloc[valid_idx])
    if X_train.shape[1] != len(columns):
        raise ValueError("A selected feature is entirely missing in training")

    # 4. FIT THE MODEL (class-balanced sample weights)
    params = {**GB_DEFAULTS, **(gb_params or {})}
    model = GradientBoostingClassifier(random_state=seed, **params)
    model.fit(X_train, y_train, sample_weight=_balanced_weight(y_train))
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    train_scores = model.decision_function(X_train)

    # 5. CLASSIFICATION METRICS (overall and per class)
    metrics = {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        # per class: electrides detected (= recall) and non-electrides correctly rejected (specificity)
        "specificity": float(recall_score(y_valid, predictions, pos_label=0, zero_division=0)),
        "non_electride_precision": float(precision_score(y_valid, predictions, pos_label=0, zero_division=0)),
        "missed_electrides": int(((y_valid == 1) & (predictions == 0)).sum()),
        "false_electrides": int(((y_valid == 0) & (predictions == 1)).sum()),
        "validation_electrides": int(y_valid.sum()), "validation_non_electrides": int((y_valid == 0).sum()),
        "confusion_matrix": confusion_matrix(y_valid, predictions, labels=[0, 1]).tolist(),
        "train_count": int(len(train_idx)), "validation_count": int(len(valid_idx)),
    }

    # 6. INDIVIDUAL FEATURE ANALYSIS (overall and per class)
    rng = np.random.default_rng(seed)
    gain = dict(zip(columns, model.feature_importances_))
    feature_analysis = {}
    for j, column in enumerate(columns):
        train_values, valid_values = X_train[:, j], X_valid[:, j]
        positive, negative = train_values[y_train == 1], train_values[y_train == 0]
        label_correlation = (float(np.corrcoef(train_values, y_train)[0, 1])
                             if np.std(train_values) > 1e-12 else None)
        univariate_auc = float(roc_auc_score(y_valid, valid_values)) if np.std(valid_values) > 1e-12 else None
        f1_drops, recall_drops, specificity_drops = [], [], []
        for _ in range(n_permutations):
            shuffled = X_valid.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            shuffled_predictions = (model.predict_proba(shuffled)[:, 1] >= 0.5).astype(int)
            f1_drops.append(metrics["f1"] - f1_score(y_valid, shuffled_predictions, zero_division=0))
            # which class loses when this feature is scrambled?
            recall_drops.append(metrics["recall"] - recall_score(y_valid, shuffled_predictions, zero_division=0))
            specificity_drops.append(metrics["specificity"]
                                     - recall_score(y_valid, shuffled_predictions, pos_label=0, zero_division=0))
        alone = _single_feature_rule(train_values, y_train, valid_values, y_valid)
        within = {}
        for name, cls in (("electride", 1), ("non_electride", 0)):
            mask = y_train == cls
            ok = mask.sum() >= 3 and np.std(train_values[mask]) > 1e-12 and np.std(train_scores[mask]) > 1e-12
            within[name] = float(np.corrcoef(train_values[mask], train_scores[mask])[0, 1]) if ok else None
        feature_analysis[column] = {
            **feature_definitions[column],
            "electride_mean": float(np.mean(positive)), "electride_std": float(np.std(positive)),
            "non_electride_mean": float(np.mean(negative)), "non_electride_std": float(np.std(negative)),
            "label_correlation": label_correlation, "univariate_auc": univariate_auc,
            "model_importance": float(gain[column]),
            "permutation_f1_importance": float(np.mean(f1_drops)), "permutation_f1_std": float(np.std(f1_drops)),
            "missing_fraction": float(X.iloc[train_idx][column].isna().mean()),
            # per class
            "permutation_electride_recall_drop": float(np.mean(recall_drops)),
            "permutation_non_electride_recall_drop": float(np.mean(specificity_drops)),
            "alone_direction": alone["direction"] if alone else None,
            "alone_threshold": alone["threshold"] if alone else None,
            "alone_electride_recall": alone["electride_recall"] if alone else None,
            "alone_non_electride_recall": alone["non_electride_recall"] if alone else None,
            "score_correlation_within_electrides": within["electride"],
            "score_correlation_within_non_electrides": within["non_electride"],
        }

    # 7. INTER-FEATURE CORRELATIONS
    correlation_matrix = pd.DataFrame(X_train, columns=columns).corr()
    correlations = {a: {b: round(float(v), 3) if pd.notna(v) else None for b, v in values.items()}
                    for a, values in correlation_matrix.to_dict().items()}

    # 8. MISCLASSIFIED STRUCTURES
    errors = [{"structure": str(df.iloc[idx]["structure"]), "true": int(y[idx]), "predicted": int(pred)}
              for idx, pred in zip(valid_idx, predictions) if int(y[idx]) != int(pred)]
    warnings = [f"{c} is selected more than once" for c, n in
                Counter(f"{f.function}__{f.feature}" for f in proposal.features).items() if n > 1]

    return {
        "features": feature_definitions, "metrics": metrics, "feature_analysis": feature_analysis,
        "correlations": correlations, "errors": errors, "warnings": warnings,
        "model": model, "imputer": imputer, "columns": columns, "gb_params": params,
        "validation_structures": [str(df.iloc[i]["structure"]) for i in valid_idx],
        "validation_labels": y_valid.tolist(), "validation_probabilities": probabilities.tolist(),
    }


def _single_feature_rule(train_values, y_train, valid_values, y_valid):
    """The feature used ALONE as a threshold rule. Direction and threshold (Youden's J) are set on the
    training rows; returns which share of each class the rule gets right on the validation rows."""
    if np.std(train_values) <= 1e-12:
        return None
    higher = train_values[y_train == 1].mean() >= train_values[y_train == 0].mean()
    sign = 1.0 if higher else -1.0
    fpr, tpr, thresholds = roc_curve(y_train, sign * train_values)
    j = int(np.argmax(tpr - fpr))
    if not np.isfinite(thresholds[j]):
        return None
    predictions = (sign * valid_values >= thresholds[j]).astype(int)
    return {"direction": "higher" if higher else "lower", "threshold": float(sign * thresholds[j]),
            "electride_recall": float(recall_score(y_valid, predictions, zero_division=0)),
            "non_electride_recall": float(recall_score(y_valid, predictions, pos_label=0, zero_division=0))}


# ============================================================ swarm-facing wrapper
def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"


class FeatureSetEvaluator:
    """Deterministic execution of agent proposals on all labeled structures in the CSV."""

    def __init__(self, catalog_path, csv_path, *, n_repeats=5, validation_fraction=0.25,
                 n_permutations=10, seed=42, max_features=5, min_coverage=0.5,
                 gb_params=None, objective="roc_auc"):
        self.catalog = load_catalog(catalog_path)
        if objective not in METRICS:
            raise ValueError(f"objective {objective!r} must be one of {METRICS}")
        self.objective = objective
        self.gb_params = {**GB_DEFAULTS, **(gb_params or {})}
        self.n_repeats, self.validation_fraction, self.n_permutations = n_repeats, validation_fraction, n_permutations
        self.seed, self.max_features, self.min_coverage = seed, max_features, min_coverage

        full = pd.read_csv(csv_path)
        if not {"structure", "label"}.issubset(full.columns):
            raise ValueError(f"{csv_path} requires 'structure' and 'label' columns")
        full = full.dropna(subset=["label"]).reset_index(drop=True)
        if not set(full["label"].unique()).issubset({0, 1}):
            raise ValueError("label must be 0/1")
        self.data = full                # every labeled structure is used; no held-out split

        self.profiles = self._profile_features()
        self.usable = {(p["function"], p["feature"]) for p in self.profiles if p["usable"]}
        if not self.usable:
            raise ValueError("No catalog feature is a usable numeric column in the CSV")
        self.formulas = _formula_labels(self.data)
        self._lock = threading.Lock()
        self.reset()

    # ---------------------------------------------------- setup helpers
    def reset(self):
        """Start a new swarm run (clears the ranking; the data is unchanged)."""
        self.history = []

    def _profile_features(self):
        profiles = []
        for rec in self.catalog:
            p = {"function": rec["function"], "feature": rec["feature"], "description": rec.get("description", ""),
                 "column": f"{rec['function']}__{rec['feature']}", "usable": False}
            if p["column"] not in self.data.columns:
                p["reason"] = "not in CSV"
            else:
                v = to_numeric(self.data[p["column"]])
                ok = v.dropna()
                p["coverage"] = float(len(ok) / len(v))
                if ok.empty:
                    p["reason"] = "not scalar (JSON/text)"
                elif ok.nunique() <= 1:
                    p["reason"] = "constant in dataset"
                elif p["coverage"] < self.min_coverage:
                    p["reason"] = f"only {p['coverage']:.0%} present"
                else:
                    p.update(usable=True, min=float(ok.min()), max=float(ok.max()), median=float(ok.median()),
                             binary=bool(set(ok.unique()) <= {0.0, 1.0}), has_zero=bool((ok == 0).any()),
                             has_negative=bool((ok < 0).any()))
            profiles.append(p)
        return profiles

    def summary(self):
        y = self.data["label"].astype(int).to_numpy()
        _, valid = train_test_split(np.arange(len(y)), test_size=self.validation_fraction,
                                    random_state=self.seed, stratify=y)
        n_vp = int(y[valid].sum())
        return (f"{len(self.data)} labeled structures ({int(y.sum())} electrides), all used for evaluation; "
                f"{len(self.usable)} of {len(self.catalog)} catalog features usable; each proposal scored on "
                f"{self.n_repeats} deterministic {1 - self.validation_fraction:.0%}/{self.validation_fraction:.0%} "
                f"train/validation splits, stratified by class: every split trains on {int(y.sum()) - n_vp} electrides + "
                f"{len(y) - int(y.sum()) - (len(valid) - n_vp)} non-electrides and validates on {n_vp} + {len(valid) - n_vp}")

    def config(self):
        return {"summary": self.summary(), "structures": self.data["structure"].tolist(),
                "usable_features": sorted(f"{f}.{g}" for f, g in self.usable), "max_features": self.max_features,
                "n_repeats": self.n_repeats, "seed": self.seed, "gb_params": self.gb_params}

    def catalog_prompt(self):
        """Feature list for the agents' system prompt (label-free statistics of all labeled structures)."""
        lines, current = ["USABLE FEATURES (use these exact function and feature names; stats over all labeled structures):"], None
        for p in self.profiles:
            if not p["usable"]:
                continue
            if p["function"] != current:
                current = p["function"]
                lines.append(f"[function: {current}]")
            if p["binary"]:
                stats = "binary 0/1"
            else:
                stats = f"range {_fmt(p['min'])} to {_fmt(p['max'])}, median {_fmt(p['median'])}"
                if p["has_zero"]:
                    stats += ", contains zeros"
            if p["coverage"] < 1:
                stats += f", {p['coverage']:.0%} present (missing -> median)"
            lines.append(f"  - {p['feature']}: {p['description']} [{stats}]")
        not_usable = [f"{p['function']}.{p['feature']} ({p['reason']})" for p in self.profiles if not p["usable"]]
        if not_usable:
            lines.append("NOT USABLE (you may propose numeric aggregations of them in suggested_new_features): "
                         + "; ".join(not_usable))
        return "\n".join(lines)

    # ---------------------------------------------------- validation
    def _validate(self, proposal):
        if not proposal.features:
            raise ValueError("Select at least one feature")
        if len(proposal.features) > self.max_features:
            raise ValueError(f"{len(proposal.features)} features selected; the maximum is {self.max_features}")
        for f in proposal.features:
            if (f.function, f.feature) not in self.usable:
                known = {(p["function"], p["feature"]): p for p in self.profiles}
                if (f.function, f.feature) in known:
                    raise ValueError(f"{f.function}.{f.feature} cannot be used: {known[(f.function, f.feature)]['reason']}")
                names = [f"{fn}.{ft}" for fn, ft in self.usable]
                close = difflib.get_close_matches(f"{f.function}.{f.feature}", names, n=3, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(f"{f.function}.{f.feature} is not a usable catalog feature.{hint}")
        return " | ".join(sorted(f"{f.function}.{f.feature}" for f in proposal.features))

    # ---------------------------------------------------- execution
    def evaluate(self, proposal, agent="?", round_number=0):
        """Validate, fit and score one feature set. Never raises for a bad proposal."""
        with self._lock:
            entry = {"id": len(self.history) + 1, "agent": agent, "round": round_number,
                     "hypothesis": proposal.hypothesis,
                     "features": [f"{f.function}.{f.feature}" for f in proposal.features],
                     "_proposal": proposal}
            try:
                key = self._validate(proposal)
                reports = [evaluate_feature_set(
                    proposal, data=self.data, seed=self.seed + r, validation_fraction=self.validation_fraction,
                    n_permutations=self.n_permutations, allowed_pairs=self.usable, gb_params=self.gb_params)
                    for r in range(self.n_repeats)]
            except Exception as exc:
                entry.update(status="invalid", error=f"{exc}")
                entry["summary"] = f"INVALID ({exc})"[:300]
                entry["feedback"] = (f"Your feature set could not be evaluated: {exc}\n"
                                     "Use only the usable features listed in the system prompt.")
                self.history.append(entry)
                return self._public(entry)
            agg = _aggregate(reports)
            first = next((e for e in self.history if e.get("key") == key and e["status"] == "ok"), None)
            entry.update(status="ok", key=key, n_features=len(proposal.features),
                         duplicate_of=(first["id"] if first else None), **agg)
            entry["progress"] = self._progress(entry)
            entry["summary"] = (f"{len(proposal.features)} features [{', '.join(entry['features'])}]: ROC-AUC "
                                f"{_fmt(agg['metrics_mean']['roc_auc'], 2)}±{_fmt(agg['metrics_std']['roc_auc'], 2)}, "
                                f"F1 {_fmt(agg['metrics_mean']['f1'], 2)}±{_fmt(agg['metrics_std']['f1'], 2)}")
            entry["feedback"] = self._feedback(entry, first)
            self.history.append(entry)
            return self._public(entry)

    def _progress(self, e):
        """Compare with the agent's best earlier feature set (its parent) on the SAME splits (paired), as facts."""
        obj = self.objective
        own = [h for h in self.history if h["agent"] == e["agent"] and h["status"] == "ok"]
        if not own:
            return {"verdict": "first", "text": "This is your first evaluated feature set; later ones are compared with your best."}
        best = max(own, key=lambda h: (h["metrics_mean"][obj], h["id"]))
        new, old = np.array(e["per_split_metrics"][obj]), np.array(best["per_split_metrics"][obj])
        d = new - old
        higher, lower = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        n, name = len(d), METRIC_NAMES[obj]
        ref = (f"its parent #{best['id']} (your best earlier set, round {best['round']}, "
               f"{name} {_fmt(best['metrics_mean'][obj], 3)})")
        now = f"{name} {_fmt(e['metrics_mean'][obj], 3)} (mean change {d.mean():+.3f})"
        if higher >= n - 1 and d.mean() > 0:
            verdict, text = "improved", f"HIGHER than {ref}: {now}, higher in {higher} of {n} splits."
        elif lower >= n - 1 and d.mean() < 0:
            verdict, text = "worse", f"LOWER than {ref}: {now}, lower in {lower} of {n} splits."
        else:
            verdict, text = "no clear change", (f"NOT CLEARLY DIFFERENT from {ref}: {now}; higher in {higher}, lower in "
                                                f"{lower} of {n} splits, i.e. within split-to-split noise.")
        return {"verdict": verdict, "text": text, "reference_id": best["id"], "delta": float(d.mean())}

    def _label(self, structure):
        stem = structure.rsplit(".", 1)[0]
        return f"{self.formulas[structure]} [{stem}]" if structure in self.formulas else stem

    @staticmethod
    def _public(entry):
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def _feedback(self, e, first):
        m, s, ps, obj = e["metrics_mean"], e["metrics_std"], e["per_split"], self.objective
        ms = lambda k, d=2: f"{_fmt(m[k], d)}±{_fmt(s[k], d)}"
        n_pos = int(self.data["label"].sum())
        roles = {c: _role(fa.get("permutation_electride_recall_drop"), fa.get("permutation_non_electride_recall_drop"))
                 for c, fa in e["feature_analysis"].items()}
        short = lambda c: e["feature_analysis"][c]["feature"]

        out = [f"Feature set: {', '.join(e['features'])}", "", "IN PLAIN WORDS:"]
        auc = m["roc_auc"]
        quality = ("very well" if auc >= 0.9 else "moderately well" if auc >= 0.8 else "poorly" if auc >= 0.65
                   else "barely better than chance")
        out.append(f"- Ranking: the model built on your features separates electrides from non-electrides {quality} "
                   f"(ROC-AUC {ms('roc_auc')}; 1.0 = perfect ranking, 0.5 = random).")
        rec, spec = m["recall"], m["specificity"]
        if spec < rec - 0.1:
            weak = "reject"
            out.append(f"- Main failure: CALLING NON-ELECTRIDES ELECTRIDES. It finds {rec:.0%} of electrides but "
                       f"rejects only {spec:.0%} of non-electrides.")
        elif rec < spec - 0.1:
            weak = "detect"
            out.append(f"- Main failure: MISSING ELECTRIDES. It rejects {spec:.0%} of non-electrides but finds only "
                       f"{rec:.0%} of electrides.")
        else:
            weak = None
            out.append(f"- Errors are similar on both classes: {rec:.0%} of electrides found, {spec:.0%} of non-electrides rejected.")
        detect = [short(c) for c, r in roles.items() if "DETECT" in r or "BOTH" in r]
        reject = [short(c) for c, r in roles.items() if "REJECT" in r or "BOTH" in r]
        idle = [short(c) for c, r in roles.items() if "little effect" in r or "HELPS" in r]
        if weak == "reject":
            out.append("- Features that currently help reject non-electrides: "
                       + (", ".join(reject) or "none of your current features does this") + ".")
        elif weak == "detect":
            out.append("- Features that currently help detect electrides: "
                       + (", ".join(detect) or "none of your current features does this") + ".")
        if idle:
            out.append("- Features with little effect, or that lower the score: " + ", ".join(idle) + ".")
        out.append(f"- Parent comparison: {e['progress']['text']}")
        if first:
            out.append(f"- Exactly the same feature set as #{first['id']} ({first['agent']}, round {first['round']}).")

        out += ["", f"NUMBERS (mean ± sd over {self.n_repeats} stratified train/validation splits of all "
                f"{len(self.data)} structures, {n_pos} electrides; every validation set holds "
                f"{ps['validation_electrides']:.0f} electrides + {ps['validation_non_electrides']:.0f} non-electrides; "
                f"objective = {METRIC_NAMES[obj]}):",
                f"- ROC-AUC {ms('roc_auc')}, F1 {ms('f1')}, accuracy {ms('accuracy')}",
                f"- At the 0.5 cutoff: electrides found {ms('recall')} ({ps['missed_electrides']:.1f} missed per split), "
                f"non-electrides rejected {ms('specificity')} ({ps['false_electrides']:.1f} wrongly called electrides "
                f"per split), precision of the electride call {ms('precision')}"]
        ops = e.get("operating_points") or {}
        if ops:
            out.append("- If the cutoff were moved: " + "; ".join(
                f"to find {int(t * 100)}% of electrides, the model would reject {_pct(v)} of non-electrides"
                for t, v in ops.items()) + ".")

        out += ["", "EACH FEATURE, BY CLASS:"]
        for column, fa in sorted(e["feature_analysis"].items(), key=lambda kv: -kv[1]["model_importance"]):
            parts = [f"share of the model's decisions {_pct(fa['model_importance'])}",
                     f"electride mean {_fmt(fa['electride_mean'])} vs non-electride mean {_fmt(fa['non_electride_mean'])}"]
            if fa.get("alone_direction") and fa.get("alone_threshold") is not None:
                sym = ">=" if fa["alone_direction"] == "higher" else "<="
                parts.append(f"used ALONE (electride if {sym} {_fmt(fa['alone_threshold'])}) it finds "
                             f"{_pct(fa['alone_electride_recall'])} of electrides and rejects "
                             f"{_pct(fa['alone_non_electride_recall'])} of non-electrides")
            parts.append(f"scrambling it lowers electride recall by {_fmt(fa.get('permutation_electride_recall_drop'), 2)} "
                         f"and non-electride rejection by {_fmt(fa.get('permutation_non_electride_recall_drop'), 2)} "
                         f"-> {roles[column]}")
            parts.append(f"correlation with the model score within electrides "
                         f"{_fmt(fa.get('score_correlation_within_electrides'), 2)}, within non-electrides "
                         f"{_fmt(fa.get('score_correlation_within_non_electrides'), 2)}")
            out.append(f"- {fa['function']}.{fa['feature']}: " + "; ".join(parts))
        strong = [f"{short(a)}-{short(b)} r={_fmt(r, 2)}" for a, row in e["correlations"].items() for b, r in row.items()
                  if a < b and r is not None and abs(r) >= 0.8]
        if strong:
            out.append("- Strongly correlated (redundant?) features: " + ", ".join(strong))

        sc = e.get("structure_scores") or {}
        hard_non = sorted((v for v in sc.values() if v["true"] == 0), key=lambda v: -v["mean_probability"])[:5]
        hard_el = sorted((v for v in sc.values() if v["true"] == 1), key=lambda v: v["mean_probability"])[:5]
        if hard_non or hard_el:
            out += ["", "STRUCTURES MOST OFTEN GOT WRONG (mean electride probability in validation; misclassified count):"]
            if hard_non:
                out.append("- Non-electrides scored most electride-like: " + "; ".join(
                    f"{self._label(v['structure'])} p={v['mean_probability']:.2f} (wrong {v['wrong']}/{v['seen']})"
                    for v in hard_non))
            if hard_el:
                out.append("- Electrides scored least electride-like: " + "; ".join(
                    f"{self._label(v['structure'])} p={v['mean_probability']:.2f} (wrong {v['wrong']}/{v['seen']})"
                    for v in hard_el))
        if e["warnings"]:
            out += ["", "Warnings: " + "; ".join(e["warnings"])]
        return "\n".join(out)

    # ---------------------------------------------------- ranking and final fit
    def leaderboard(self, metric=None, top=None):
        """Unique evaluated feature sets ranked by the validation mean of `metric` (default: the objective)."""
        metric = metric or self.objective
        second = "f1" if metric == "roc_auc" else "roc_auc"
        seen, rows = set(), []
        for e in sorted((e for e in self.history if e["status"] == "ok"),
                        key=lambda e: (-e["metrics_mean"][metric], -e["metrics_mean"][second], e["id"])):
            if e["key"] in seen:
                continue
            seen.add(e["key"])
            rows.append({k: e[k] for k in ("id", "agent", "round", "features", "n_features",
                                           "metrics_mean", "metrics_std", "hypothesis", "feature_analysis")})
        return rows[:top] if top else rows

    def select_best(self, metric=None):
        """The top of the ranking: highest validation mean of `metric` (default: the objective)."""
        metric = metric or self.objective
        board = self.leaderboard(metric)
        if not board:
            return None
        return {**board[0], "selection_rule": f"highest validation {METRIC_NAMES[metric]} (mean over {self.n_repeats} splits)"}

    def final_fit(self, entry, model_path=None):
        """Refit the selected feature set on ALL labeled structures, for export to an external test set.
        The metrics returned here are in-sample (training) metrics and therefore optimistic."""
        proposal = next(e["_proposal"] for e in self.history if e["id"] == entry["id"])
        everything = np.arange(len(self.data))
        with self._lock:
            report = evaluate_feature_set(proposal, data=self.data, split=(everything, everything),
                                          seed=self.seed, n_permutations=self.n_permutations,
                                          allowed_pairs=self.usable, gb_params=self.gb_params)
        report["errors"] = [{**x, "formula": self.formulas.get(x["structure"])} for x in report["errors"]]
        medians = {c: float(to_numeric(self.data[c]).median()) for c in report["columns"]}
        if model_path:
            joblib.dump({"model": report["model"], "imputer": report["imputer"], "columns": report["columns"],
                         "medians": medians, "features": entry["features"]}, model_path)
        return {"status": "ok", "id": entry["id"], "n_structures": len(self.data),
                "features": entry["features"], "columns": report["columns"], "gb_params": report["gb_params"],
                "importances": {c: float(v["model_importance"]) for c, v in report["feature_analysis"].items()},
                "imputation_medians": medians, "model_file": model_path,
                "in_sample_metrics": report["metrics"],
                "cross_validated_metrics": {"mean": entry["metrics_mean"], "std": entry["metrics_std"]},
                "note": "Model fitted on all labeled structures; probability >= 0.5 means electride. "
                        "Missing feature values are imputed with the median of these structures."}


# ============================================================ apply an exported model
def score_dataset(model=None, data_csv="electride_features_dataset.csv", out_csv="outputs/final_model_scores.csv",
                  outputs_dir="outputs"):
    """Score any feature CSV with the same function__feature columns using the exported model.

    model: the final_fit dict returned by the swarm, a path to outputs/final_model_*.joblib, or None for the
    newest one in outputs_dir. Returns (scores DataFrame, metrics dict or None)."""
    if isinstance(model, dict):
        source = model.get("model_file")
    else:
        source = model or sorted(glob.glob(f"{outputs_dir}/final_model_*.joblib"))[-1]
    if not source or not os.path.exists(source):
        raise FileNotFoundError(f"No exported model file found ({source!r}); run the swarm first.")
    bundle = joblib.load(source)
    print(f"Model from {source}:\n  features: {', '.join(bundle['features'])}")

    df = pd.read_csv(data_csv)
    values = []
    for column in bundle["columns"]:
        if column not in df.columns:
            raise KeyError(f"{data_csv} has no column {column}")
        v = to_numeric(df[column])
        if v.isna().any():
            print(f"  {column}: {int(v.isna().sum())} missing -> training median {bundle['medians'][column]:.4g}")
        values.append(v.fillna(bundle["medians"][column]).to_numpy())
    X = bundle["imputer"].transform(np.column_stack(values))
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    scores = pd.DataFrame({"structure": df["structure"] if "structure" in df else df.index,
                           "probability": probabilities, "predicted": (probabilities >= 0.5).astype(int)})
    metrics = None
    if "label" in df:
        scores["label"] = df["label"]
        m = df["label"].notna().to_numpy()
        y, pred, p = df["label"][m].astype(int), scores["predicted"][m].astype(int), probabilities[m]
        metrics = {"n": int(m.sum()), "electrides": int(y.sum()), "accuracy": accuracy_score(y, pred),
                   "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
                   "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
                   "f1": f1_score(y, pred, zero_division=0),
                   "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else float("nan")}
        print("Metrics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
        print("Confusion matrix [[TN, FP], [FN, TP]]:", confusion_matrix(y, pred, labels=[0, 1]).tolist())
    if out_csv:
        scores.to_csv(out_csv, index=False)
        print(f"Saved {len(scores)} scores to {out_csv}")
    return scores, metrics


# ============================================================ helpers
def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"

def _role(electride_drop, non_electride_drop, tol=0.05):
    """Which class a feature serves in the model, from the class-specific permutation drops."""
    if electride_drop is None or non_electride_drop is None:
        return "n/a"
    if electride_drop > tol and non_electride_drop > tol:
        return "needed for BOTH classes"
    if electride_drop > tol:
        return "mainly needed to DETECT electrides"
    if non_electride_drop > tol:
        return "mainly needed to REJECT non-electrides"
    if electride_drop < -tol or non_electride_drop < -tol:
        return "scrambling it HELPS one class: likely hurting the model"
    return "little effect in this model (redundant with the other features?)"

def _mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None

def _operating_points(labels, probabilities, targets=(0.8, 0.9, 0.95)):
    """From the ROC curve: share of non-electrides rejected at the cutoff that finds each target share of electrides."""
    if len(set(labels)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return {t: float(1 - fpr[np.argmax(tpr >= t - 1e-12)]) for t in targets}

def _aggregate(reports):
    metrics_mean = {k: float(np.mean([r["metrics"][k] for r in reports])) for k in METRICS}
    metrics_std = {k: float(np.std([r["metrics"][k] for r in reports])) for k in METRICS}
    fields = ("electride_mean", "non_electride_mean", "label_correlation", "univariate_auc", "model_importance",
              "permutation_f1_importance", "missing_fraction",
              "permutation_electride_recall_drop", "permutation_non_electride_recall_drop", "alone_threshold",
              "alone_electride_recall", "alone_non_electride_recall",
              "score_correlation_within_electrides", "score_correlation_within_non_electrides")
    feature_analysis = {c: {**{k: reports[0]["feature_analysis"][c][k] for k in
                               ("function", "feature", "csv_column", "alone_direction")},
                            **{k: _mean([r["feature_analysis"][c][k] for r in reports]) for k in fields}}
                        for c in reports[0]["feature_analysis"]}
    correlations = {a: {b: _mean([r["correlations"][a][b] for r in reports]) for b in row}
                    for a, row in reports[0]["correlations"].items()}
    seen, wrong, truth = Counter(), Counter(), {}
    for r in reports:
        seen.update(r["validation_structures"])
        for x in r["errors"]:
            wrong[x["structure"]] += 1
            truth[x["structure"]] = x["true"]
    misclassified = [{"structure": s, "true": truth[s], "wrong": n, "seen": seen[s]}
                     for s, n in sorted(wrong.items(), key=lambda kv: (-kv[1], -kv[1] / seen[kv[0]], kv[0]))]
    per_split = {k: float(np.mean([r["metrics"][k] for r in reports]))
                 for k in ("missed_electrides", "false_electrides", "validation_electrides", "validation_non_electrides")}
    roc = {"labels": [v for r in reports for v in r["validation_labels"]],
           "probabilities": [v for r in reports for v in r["validation_probabilities"]]}
    per_split_metrics = {k: [float(r["metrics"][k]) for r in reports] for k in METRICS}
    structure_scores = {}
    for r in reports:
        for st, yv, pv in zip(r["validation_structures"], r["validation_labels"], r["validation_probabilities"]):
            v = structure_scores.setdefault(st, {"structure": st, "true": int(yv), "probabilities": []})
            v["probabilities"].append(float(pv))
    for v in structure_scores.values():
        p = np.array(v.pop("probabilities"))
        v.update(mean_probability=float(p.mean()), seen=int(len(p)),
                 wrong=int(((p >= 0.5).astype(int) != v["true"]).sum()))
    return {"metrics_mean": metrics_mean, "metrics_std": metrics_std, "per_split": per_split, "roc": roc,
            "per_split_metrics": per_split_metrics, "structure_scores": structure_scores,
            "operating_points": _operating_points(roc["labels"], roc["probabilities"]),
            "feature_analysis": feature_analysis, "correlations": correlations, "misclassified": misclassified,
            "warnings": reports[0]["warnings"]}

def _formula_labels(df):
    """structure -> reduced formula (from structure_summary composition columns), for readable feedback."""
    col = "structure_summary__composition_per_cell"
    element_cols = [c for c in df.columns if c.startswith(col + "__")]
    out = {}
    for _, row in df.iterrows():
        comp = None
        if col in df.columns:
            try:
                raw = json.loads(row[col]) if isinstance(row[col], str) else None
                comp = dict(raw) if isinstance(raw, (dict, list)) else None
            except (ValueError, TypeError):
                comp = None
        if comp is None and element_cols:
            comp = {c[len(col) + 2:]: row[c] for c in element_cols if pd.notna(row[c]) and row[c]}
        try:
            counts = {el: float(n) for el, n in comp.items() if float(n) > 0}
            if counts and all(v.is_integer() for v in counts.values()):
                g = math.gcd(*(int(v) for v in counts.values()))
                out[str(row["structure"])] = "".join(f"{el}{'' if int(v) // g == 1 else int(v) // g}"
                                                     for el, v in counts.items())
        except (AttributeError, TypeError, ValueError):
            pass
    return out
