"""
================================================================================
TEST RUNNER — 8 datasets × FraudModelValidatorV3
================================================================================
Lance une validation complète sur chaque dataset et produit un tableau
comparatif des métriques clés + alertes attendues.

Usage :
    python run_all_tests.py

    # Un seul dataset :
    python run_all_tests.py --ds DS05_smurfing_benford

Prérequis :
    pip install scikit-learn pandas numpy scipy statsmodels shap networkx antropy
    Les fichiers DS0*.csv doivent être dans le même dossier que ce script,
    ou dans le sous-dossier datasets/.
================================================================================
"""

import argparse
import os
import sys
import time
import warnings
import textwrap

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Localiser les datasets ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "datasets")]

def find_csv(name):
    for d in DATA_DIRS:
        p = os.path.join(d, f"{name}.csv")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Introuvable : {name}.csv (cherché dans {DATA_DIRS})")

# ── Import du framework ───────────────────────────────────────────────────────
try:
    from Fraud_Model_Validator_v3d import FraudModelValidatorV3
except ImportError:
    sys.exit(
        "❌  Impossible d'importer FraudModelValidatorV3.\n"
        "   Assurez-vous que Fraud_Model_Validator_v3d.py est dans le même dossier."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Catalogue des datasets
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = [
    {
        "id":   "DS01_basic_fraud",
        "name": "DS01 — Fraude basique",
        "desc": "Baseline : séparation parfaite entre légitimes et fraudes.",
        "expected": {
            "auc":  "> 0.97",
            "ks":   "> 0.90",
            "benford": "Conforme",
            "ultra_risk": "LOW",
        },
        "focus": ["discriminante"],
    },
    {
        "id":   "DS02_nocturnal_fraud",
        "name": "DS02 — Fraude nocturne",
        "desc": "Signal unique : heure de la journée. Montants identiques.",
        "expected": {
            "auc":  "~0.70–0.80",
            "ks":   "modéré",
            "benford": "Conforme",
            "ultra_risk": "LOW/MEDIUM",
        },
        "focus": ["discriminante", "threshold_optimization"],
    },
    {
        "id":   "DS03_extreme_imbalance",
        "name": "DS03 — Déséquilibre extrême",
        "desc": "0.2% de fraude. F1 et MCC très challengeants.",
        "expected": {
            "auc":  "> 0.95",
            "f1":   "< 0.70 au seuil 0.5",
            "benford": "Conforme",
        },
        "focus": ["discriminante", "cost_analysis"],
    },
    {
        "id":   "DS04_concept_drift",
        "name": "DS04 — Concept drift",
        "desc": "Le profil de fraude change à mi-parcours.",
        "expected": {
            "auc_wf": "décroissant sur les derniers folds",
            "bocd":   "changement de régime détecté",
        },
        "focus": ["walk_forward", "ultra"],
    },
    {
        "id":   "DS05_smurfing_benford",
        "name": "DS05 — Smurfing / Benford",
        "desc": "Montants juste sous les seuils KYC (9999€, 4999€, 2999€).",
        "expected": {
            "benford": "NON conforme — violation chiffres 9, 4, 2",
            "mad":     "> 0.010",
            "chi2_p":  "< 0.05",
        },
        "focus": ["benford"],
    },
    {
        "id":   "DS06_ring_network",
        "name": "DS06 — Réseau organisé",
        "desc": "Anneau de 20 comptes complices (IDs 90001–90020).",
        "expected": {
            "graph_hubs": "comptes 9000x détectés",
            "ultra_risk": "LOW à MEDIUM",
        },
        "focus": ["graph"],
    },
    {
        "id":   "DS07_synthetic_rmt",
        "name": "DS07 — Données synthétiques",
        "desc": "Features indépendantes uniformes → corrélations non naturelles.",
        "expected": {
            "rmt_pvalue": "< 0.05",
            "ultra_risk": "MEDIUM à HIGH",
            "auc":        "~0.50 (aléatoire)",
        },
        "focus": ["ultra"],
    },
    {
        "id":   "DS08_label_noise",
        "name": "DS08 — Label noise",
        "desc": "10% des labels inversés — attaque adversariale.",
        "expected": {
            "auc":  "dégradé ~0.80–0.90",
            "mcc":  "< 0.75",
            "brier": "dégradé",
        },
        "focus": ["discriminante", "calibration"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Fonction de test pour un dataset
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "transaction_amount", "transaction_velocity_1h", "merchant_risk_score",
    "customer_age_days", "is_international", "hour_of_day", "device_risk_score",
]

def run_dataset(ds_spec, run_ultra=True, run_graph=True, run_benford=True,
                run_wf=False, verbose=True):
    t0 = time.time()
    ds_id   = ds_spec["id"]
    ds_name = ds_spec["name"]

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {ds_name}")
        print(f"  {ds_spec['desc']}")
        print(f"{'─'*70}")

    # ── Chargement ────────────────────────────────────────────────────────────
    df = pd.read_csv(find_csv(ds_id), parse_dates=["transaction_date"])
    n_total = len(df)
    n_fraud = df["is_fraud"].sum()
    if verbose:
        print(f"  Données : {n_total:,} transactions | {n_fraud:,} fraudes ({100*n_fraud/n_total:.2f}%)")

    # ── Train / Test split ────────────────────────────────────────────────────
    X = df[FEATURE_COLS]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    model  = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    )
    model.fit(scaler.fit_transform(X_train), y_train)
    y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    y_true  = y_test.values

    def predict_fn(X_in):
        return model.predict_proba(scaler.transform(X_in))[:, 1]

    # ── Validateur ────────────────────────────────────────────────────────────
    validator = FraudModelValidatorV3(
        model_name=ds_name, cost_fn=5000, cost_fp=100
    )

    # ── Performance ───────────────────────────────────────────────────────────
    metrics = validator.calculate_performance_metrics(y_true, y_proba, bootstrap=False)
    if verbose:
        print(f"\n  ► Performance")
        print(f"    AUC-ROC   : {metrics.get('auc_roc', 0):.4f}")
        print(f"    GINI      : {metrics.get('gini', 0):.4f}")
        print(f"    KS        : {metrics.get('ks_statistic', 0):.4f}")
        print(f"    F1        : {metrics.get('f1_score', 0):.4f}")
        print(f"    MCC       : {metrics.get('mcc', 0):.4f}")

    # ── Calibration ───────────────────────────────────────────────────────────
    cal = validator.calibration_analysis(y_true, y_proba)
    if verbose:
        print(f"\n  ► Calibration")
        print(f"    Brier     : {cal.get('brier_score', 0):.4f}")
        print(f"    ECE       : {cal.get('ece', 0):.4f}")
        print(f"    Log-Loss  : {cal.get('log_loss', 0):.4f}")

    # ── Benford ───────────────────────────────────────────────────────────────
    benford_result = {}
    if run_benford:
        try:
            benford_result = validator.benford_law_analysis(
                df, amount_col="transaction_amount"
            )
            if verbose:
                print(f"\n  ► Benford's Law")
                print(f"    MAD       : {benford_result.get('mad', 0):.5f}")
                print(f"    χ² p-val  : {benford_result.get('chi2_pvalue', 0):.5f}")
                print(f"    Conformité: {benford_result.get('conformity', 'N/A')}")
        except Exception as e:
            if verbose:
                print(f"\n  ► Benford : erreur — {e}")

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph_result = {}
    if run_graph:
        try:
            df_test_graph = X_test.copy()
            df_test_graph["is_fraud"] = y_true
            df_test_graph["sender_id"]   = df["sender_id"].iloc[X_test.index].values
            df_test_graph["receiver_id"] = df["receiver_id"].iloc[X_test.index].values
            df_test_graph["transaction_amount"] = df["transaction_amount"].iloc[X_test.index].values
            graph_result = validator.graph_anomaly_detection(df_test_graph)
            if verbose and graph_result:
                print(f"\n  ► Graph Anomaly")
                print(f"    Nœuds     : {graph_result.get('n_nodes', 0):,}")
                print(f"    Arêtes    : {graph_result.get('n_edges', 0):,}")
        except Exception as e:
            if verbose:
                print(f"\n  ► Graph : erreur — {e}")

    # ── Ultra-Scientific ─────────────────────────────────────────────────────
    ultra_result = {}
    if run_ultra:
        try:
            df_test_ultra = X_test.copy()
            df_test_ultra["transaction_date"] = df["transaction_date"].iloc[X_test.index].values
            df_test_ultra["score"] = y_proba
            ultra_result = validator.ultra_scientific_detection(
                df_test_ultra, FEATURE_COLS, time_col="transaction_date"
            )
            if verbose:
                print(f"\n  ► Ultra-Scientific Detection")
                print(f"    Score     : {ultra_result.get('global_ultra_anomaly_score', 0):.4f}")
                print(f"    Risque    : {ultra_result.get('ultra_risk_level', 'N/A')}")
                print(f"    MF-DFA    : {ultra_result.get('mfdfa_multifractal_width', 0):.4f}")
                print(f"    Transfer Entropy : {ultra_result.get('transfer_entropy_mean', 0):.4f}")
                print(f"    RMT p-val : {ultra_result.get('rmt_pvalue', 0):.4f}")
                print(f"    ESN res.  : {ultra_result.get('esn_residual_mean', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"\n  ► Ultra-Scientific : erreur — {e}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    wf_result = {}
    if run_wf and "walk_forward" in ds_spec.get("focus", []):
        try:
            wf_result = validator.walk_forward_backtest(
                df, model=model,
                feature_cols=FEATURE_COLS,
                target_col="is_fraud",
                date_col="transaction_date",
                n_splits=4,
            )
            if verbose:
                print(f"\n  ► Walk-Forward Backtest")
                print(f"    AUC moyen : {wf_result.get('mean_auc', 0):.4f}")
                print(f"    AUC std   : {wf_result.get('std_auc', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"\n  ► Walk-Forward : erreur — {e}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  ✓ Terminé en {elapsed:.1f}s")
        # Alertes attendues
        print(f"\n  ℹ ATTENDU : ", end="")
        for k, v in ds_spec["expected"].items():
            print(f"{k}={v}  ", end="")
        print()

    return {
        "id":      ds_id,
        "name":    ds_name,
        "n_total": n_total,
        "n_fraud": int(n_fraud),
        "fraud_pct": round(100 * n_fraud / n_total, 2),
        "auc":     round(metrics.get("auc_roc", 0), 4),
        "gini":    round(metrics.get("gini", 0), 4),
        "ks":      round(metrics.get("ks_statistic", 0), 4),
        "f1":      round(metrics.get("f1_score", 0), 4),
        "mcc":     round(metrics.get("mcc", 0), 4),
        "brier":   round(cal.get("brier_score", 0), 4),
        "ece":     round(cal.get("ece", 0), 4),
        "benford_conformity": benford_result.get("conformity", "N/A"),
        "benford_mad":        round(benford_result.get("mad", 0), 5),
        "ultra_score":        round(ultra_result.get("global_ultra_anomaly_score", 0), 4),
        "ultra_risk":         ultra_result.get("ultra_risk_level", "N/A"),
        "rmt_pvalue":         round(ultra_result.get("rmt_pvalue", 0), 4),
        "elapsed_s":          round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test runner — FraudModelValidatorV3")
    parser.add_argument("--ds", default=None, help="ID d'un seul dataset (ex: DS05_smurfing_benford)")
    parser.add_argument("--no-ultra",   action="store_true", help="Désactiver l'ultra-scientific detection")
    parser.add_argument("--no-graph",   action="store_true", help="Désactiver le graph analysis")
    parser.add_argument("--no-benford", action="store_true", help="Désactiver Benford")
    parser.add_argument("--walk-forward", action="store_true", help="Activer le walk-forward (lent)")
    args = parser.parse_args()

    datasets = DATASETS
    if args.ds:
        datasets = [d for d in DATASETS if args.ds.lower() in d["id"].lower()]
        if not datasets:
            sys.exit(f"Dataset non trouvé : {args.ds}")

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   FRAUD DATASET TEST RUNNER — FraudModelValidatorV3                 ║
║   8 scénarios progressifs | SR 11-7 · ECB TRIM · AMLD6              ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    results = []
    for ds in datasets:
        try:
            r = run_dataset(
                ds,
                run_ultra=not args.no_ultra,
                run_graph=not args.no_graph,
                run_benford=not args.no_benford,
                run_wf=args.walk_forward,
            )
            results.append(r)
        except FileNotFoundError as e:
            print(f"\n  ⚠ {e} — Skipped")
        except Exception as e:
            print(f"\n  ✗ Erreur inattendue sur {ds['id']} : {e}")
            import traceback; traceback.print_exc()

    if not results:
        return

    # ── Tableau de synthèse ───────────────────────────────────────────────────
    df_res = pd.DataFrame(results)
    print(f"\n\n{'═'*100}")
    print("  TABLEAU DE SYNTHÈSE COMPARATIF")
    print(f"{'═'*100}")
    print(f"{'Dataset':<32} {'Fraude%':>7} {'AUC':>7} {'KS':>7} {'F1':>7} {'MCC':>7} "
          f"{'Brier':>7} {'Benford':>12} {'Ultra':>6} {'Risk':<8} {'RMT p':>7} {'t(s)':>6}")
    print("─" * 100)
    for _, r in df_res.iterrows():
        print(
            f"{r['name']:<32} {r['fraud_pct']:>6.2f}% {r['auc']:>7.4f} {r['ks']:>7.4f} "
            f"{r['f1']:>7.4f} {r['mcc']:>7.4f} {r['brier']:>7.4f} "
            f"{str(r['benford_conformity']):>12} {r['ultra_score']:>6.4f} "
            f"{str(r['ultra_risk']):<8} {r['rmt_pvalue']:>7.4f} {r['elapsed_s']:>5.1f}s"
        )
    print(f"{'═'*100}")

    # ── Sauvegarde des résultats ──────────────────────────────────────────────
    out_csv = "test_results_summary.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n  Résultats sauvegardés → {out_csv}")

    # ── Points d'attention ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  POINTS D'ATTENTION AUTOMATIQUES")
    print(f"{'─'*70}")

    alerts = []
    for _, r in df_res.iterrows():
        name = r["name"]
        if r["auc"] < 0.70:
            alerts.append(f"  🔴 {name} : AUC faible ({r['auc']:.4f}) — modèle peu discriminant")
        if r["ultra_risk"] == "HIGH":
            alerts.append(f"  🔴 {name} : Ultra Risk HIGH — anomalie scientifique critique")
        if r["ultra_risk"] == "MEDIUM":
            alerts.append(f"  🟡 {name} : Ultra Risk MEDIUM — surveillance recommandée")
        if str(r["benford_conformity"]) not in ("Conforme", "N/A"):
            alerts.append(f"  🟡 {name} : Benford non conforme (MAD={r['benford_mad']:.5f})")
        if r["rmt_pvalue"] < 0.05:
            alerts.append(f"  🔴 {name} : RMT p={r['rmt_pvalue']:.4f} — corrélations non naturelles")
        if r["mcc"] < 0.50 and r["fraud_pct"] > 0.5:
            alerts.append(f"  🟡 {name} : MCC={r['mcc']:.4f} — performance dégradée")

    if alerts:
        for a in alerts:
            print(a)
    else:
        print("  ✅ Aucune alerte majeure détectée.")
    print()


if __name__ == "__main__":
    main()
