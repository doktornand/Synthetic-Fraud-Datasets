"""
================================================================================
FRAUD DATASET GENERATOR — 8 scénarios progressifs
================================================================================
Génère 8 datasets CSV prêts à l'emploi pour tester FraudModelValidatorV3.

Chaque dataset respecte le schéma attendu par le framework :
  - transaction_amount    : montant (float > 0)
  - transaction_velocity_1h : nb tx dernière heure (int)
  - merchant_risk_score   : score risque commerçant [0,1]
  - customer_age_days     : ancienneté client (jours)
  - is_international      : 0/1
  - hour_of_day           : heure 0-23
  - device_risk_score     : score risque device [0,1]
  - is_fraud              : label binaire 0/1
  - transaction_date      : datetime
  - sender_id             : identifiant émetteur
  - receiver_id           : identifiant destinataire

SCÉNARIOS (complexité croissante) :
  DS01 — Fraude basique            : séparation parfaite, très facile
  DS02 — Fraude nocturne           : signal temporel unique
  DS03 — Classe très déséquilibrée : 0.2% de fraude
  DS04 — Fraude graduelle (drift)  : le profil évolue dans le temps
  DS05 — Smurfing (Benford)        : montants juste en dessous de seuils
  DS06 — Réseau organisé (graph)   : anneau de comptes liés
  DS07 — Données synthétiques      : corrélations non naturelles (test RMT)
  DS08 — Adversarial / label noise : 10% de labels bruités
================================================================================
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
OUT = "/home/claude/datasets"
os.makedirs(OUT, exist_ok=True)

N_LEGIT  = 9750   # taux fraude standard ~2.5%
N_FRAUD  = 250
N_TOTAL  = N_LEGIT + N_FRAUD

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def base_legit(n, rng=RNG):
    return {
        'transaction_amount':      rng.lognormal(4.0, 1.5, n),
        'transaction_velocity_1h': rng.poisson(2, n),
        'merchant_risk_score':     rng.beta(2, 5, n),
        'customer_age_days':       rng.exponential(365, n),
        'is_international':        rng.binomial(1, 0.05, n),
        'hour_of_day':             rng.integers(8, 20, n),
        'device_risk_score':       rng.beta(1, 5, n),
        'is_fraud': np.zeros(n, int),
    }

def base_fraud(n, rng=RNG):
    return {
        'transaction_amount':      rng.lognormal(6.5, 1.0, n),
        'transaction_velocity_1h': rng.poisson(10, n),
        'merchant_risk_score':     rng.beta(6, 2, n),
        'customer_age_days':       rng.exponential(20, n),
        'is_international':        rng.binomial(1, 0.5, n),
        'hour_of_day':             rng.choice(list(range(0,6)) + list(range(22,24)), n),
        'device_risk_score':       rng.beta(5, 1, n),
        'is_fraud': np.ones(n, int),
    }

def finalize(legit_dict, fraud_dict, start_date='2024-01-01', freq='2min',
             n_senders=5000, n_receivers=3000, rng=RNG):
    df = pd.concat([pd.DataFrame(legit_dict), pd.DataFrame(fraud_dict)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    df['transaction_date'] = pd.date_range(start_date, periods=n, freq=freq)
    df['sender_id']   = rng.integers(1, n_senders,  n)
    df['receiver_id'] = rng.integers(1, n_receivers, n)
    return df

def save(df, name, desc):
    path = f"{OUT}/{name}.csv"
    df.to_csv(path, index=False)
    n_fraud = df['is_fraud'].sum()
    pct = 100 * n_fraud / len(df)
    print(f"  ✓ {name}.csv  —  {len(df):,} tx | {n_fraud:,} fraudes ({pct:.2f}%)  | {desc}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# DS01 — Fraude basique
# Séparation très nette entre légitimes et fraudes. Toutes les features
# discriminent fortement. Sert de référence/baseline.
# Attendu : AUC > 0.97, KS > 0.90
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS01] Fraude basique...")
l = base_legit(N_LEGIT)
f = base_fraud(N_FRAUD)
ds01 = finalize(l, f)
save(ds01, "DS01_basic_fraud", "Séparation parfaite — baseline")

# ─────────────────────────────────────────────────────────────────────────────
# DS02 — Fraude nocturne (signal temporel unique)
# Les fraudes ont le même montant que les légitimes, mais surviennent
# uniquement entre 00h et 05h. Une seule feature discrimine.
# Attendu : AUC modéré (~0.75), sensible au seuil, test BOCD utile
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS02] Fraude nocturne...")
l2 = base_legit(N_LEGIT)
f2 = base_fraud(N_FRAUD)
# Override : montants similaires, seule l'heure discrimine
f2['transaction_amount']      = RNG.lognormal(4.0, 1.5, N_FRAUD)   # même que légitimes
f2['transaction_velocity_1h'] = RNG.poisson(2, N_FRAUD)             # idem
f2['merchant_risk_score']     = RNG.beta(2, 5, N_FRAUD)             # idem
f2['customer_age_days']       = RNG.exponential(365, N_FRAUD)       # idem
f2['is_international']        = RNG.binomial(1, 0.05, N_FRAUD)      # idem
f2['device_risk_score']       = RNG.beta(1, 5, N_FRAUD)             # idem
f2['hour_of_day']             = RNG.integers(0, 5, N_FRAUD)         # ← seul signal
ds02 = finalize(l2, f2)
save(ds02, "DS02_nocturnal_fraud", "Signal temporel unique (heure)")

# ─────────────────────────────────────────────────────────────────────────────
# DS03 — Classe extrêmement déséquilibrée (0.2% de fraude)
# Simule un contexte réel où la fraude est très rare.
# Attendu : Precision-Recall challengeant, F1 faible si seuil = 0.5
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS03] Classe très déséquilibrée...")
N_LEGIT3 = 9980
N_FRAUD3  = 20    # 0.2%
l3 = base_legit(N_LEGIT3)
f3 = base_fraud(N_FRAUD3)
ds03 = finalize(l3, f3)
save(ds03, "DS03_extreme_imbalance", "0.2% fraude — Precision-Recall challengeant")

# ─────────────────────────────────────────────────────────────────────────────
# DS04 — Concept drift (fraude graduelle)
# Le profil des fraudes évolue dans le temps : au début, fraudes classiques
# (montants élevés) ; à mi-parcours, la fraude migre vers de petits montants
# nocturnes. Teste walk-forward et BOCD.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS04] Concept drift...")
# Phase 1 (première moitié) : fraude classique
l4a = base_legit(N_LEGIT // 2)
f4a = base_fraud(N_FRAUD // 2)

# Phase 2 (deuxième moitié) : fraude migre vers petits montants nocturnes
l4b = base_legit(N_LEGIT // 2)
f4b = base_fraud(N_FRAUD // 2)
f4b['transaction_amount']  = RNG.lognormal(3.5, 0.5, N_FRAUD // 2)   # petits montants
f4b['hour_of_day']         = RNG.integers(0, 4, N_FRAUD // 2)
f4b['merchant_risk_score'] = RNG.beta(2, 4, N_FRAUD // 2)             # moins marqué

# Concaténer en ordre chronologique (pas de shuffle global !)
df4a = pd.DataFrame({**l4a, **{'is_fraud': np.zeros(N_LEGIT//2, int)}})
df4b_fraud = pd.DataFrame({**f4a, **{'is_fraud': np.ones(N_FRAUD//2, int)}})
phase1 = pd.concat([df4a, df4b_fraud]).sample(frac=1, random_state=1).reset_index(drop=True)

df4c = pd.DataFrame({**l4b, **{'is_fraud': np.zeros(N_LEGIT//2, int)}})
df4d = pd.DataFrame({**f4b, **{'is_fraud': np.ones(N_FRAUD//2, int)}})
phase2 = pd.concat([df4c, df4d]).sample(frac=1, random_state=2).reset_index(drop=True)

ds04 = pd.concat([phase1, phase2], ignore_index=True)
ds04['transaction_date'] = pd.date_range('2024-01-01', periods=len(ds04), freq='2min')
ds04['sender_id']   = RNG.integers(1, 5000, len(ds04))
ds04['receiver_id'] = RNG.integers(1, 3000, len(ds04))
save(ds04, "DS04_concept_drift", "Profil de fraude qui évolue dans le temps")

# ─────────────────────────────────────────────────────────────────────────────
# DS05 — Smurfing / Benford violation
# Les fraudeurs fractionnent leurs transactions juste en dessous de seuils
# réglementaires : 9 999€, 4 999€, 2 999€ (sous les seuils KYC).
# La loi de Benford sera violée : surreprésentation des chiffres 9, 4, 2.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS05] Smurfing / Benford violation...")
l5 = base_legit(N_LEGIT)

# Fraudes : montants concentrés juste sous des seuils
thresholds = [9999, 4999, 2999, 999]
fraud_amounts = RNG.choice(thresholds, N_FRAUD) - RNG.uniform(0.50, 50.0, N_FRAUD)
fraud_amounts = np.clip(fraud_amounts, 10, None)

f5 = base_fraud(N_FRAUD)
f5['transaction_amount']      = fraud_amounts
f5['transaction_velocity_1h'] = RNG.poisson(5, N_FRAUD)    # fragmentation → vitesse modérée
f5['merchant_risk_score']     = RNG.beta(3, 3, N_FRAUD)    # commerçants moins suspects
f5['hour_of_day']             = RNG.integers(9, 18, N_FRAUD)  # heures ouvrées pour passer inaperçu

ds05 = finalize(l5, f5)
save(ds05, "DS05_smurfing_benford", "Smurfing juste sous les seuils KYC (Benford)")

# ─────────────────────────────────────────────────────────────────────────────
# DS06 — Réseau de fraude organisée (graph anomaly)
# Un petit groupe de 20 comptes se transactent mutuellement en boucle.
# Ces comptes constituent un anneau fermé de blanchiment.
# Attendu : detectable par graph_anomaly_detection (hubs, cliques)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS06] Réseau organisé (ring fraud)...")
l6 = base_legit(N_LEGIT)
f6 = base_fraud(N_FRAUD)

# Anneau de 20 comptes complices
ring_accounts = list(range(90001, 90021))    # IDs uniques hors plage normale

ds06 = finalize(l6, f6, n_senders=5000, n_receivers=3000)

# Réassigner sender/receiver des fraudes vers l'anneau
fraud_mask = ds06['is_fraud'] == 1
n_f = fraud_mask.sum()
ds06.loc[fraud_mask, 'sender_id']   = RNG.choice(ring_accounts, n_f)
ds06.loc[fraud_mask, 'receiver_id'] = RNG.choice(ring_accounts, n_f)
# Quelques transactions légitimes impliquent aussi ces comptes (bruit)
noise_idx = RNG.choice(ds06[~fraud_mask].index, 30, replace=False)
ds06.loc[noise_idx, 'receiver_id'] = RNG.choice(ring_accounts, 30)

save(ds06, "DS06_ring_network", "Anneau de 20 comptes complices (graph anomaly)")

# ─────────────────────────────────────────────────────────────────────────────
# DS07 — Données synthétiques / RMT trap
# Toutes les features sont générées indépendamment par des lois uniformes.
# Les corrélations inter-features sont quasi-nulles et régulières,
# ce qui viole la distribution de Marchenko-Pastur attendue sur données réelles.
# Attendu : RMT p-value très faible (< 0.05) → alerte ultra-scientifique
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS07] Données synthétiques (RMT trap)...")
n = N_TOTAL
# Toutes les features : distributions uniformes indépendantes (aucune corrélation naturelle)
ds07 = pd.DataFrame({
    'transaction_amount':      RNG.uniform(10, 10000, n),
    'transaction_velocity_1h': RNG.integers(0, 15, n),
    'merchant_risk_score':     RNG.uniform(0, 1, n),
    'customer_age_days':       RNG.uniform(1, 3650, n),
    'is_international':        RNG.integers(0, 2, n),
    'hour_of_day':             RNG.integers(0, 24, n),
    'device_risk_score':       RNG.uniform(0, 1, n),
})
# Labels artificiellement attribués : 2.5% de fraude au hasard
ds07['is_fraud'] = 0
fraud_idx = RNG.choice(n, N_FRAUD, replace=False)
ds07.loc[fraud_idx, 'is_fraud'] = 1

ds07 = ds07.sample(frac=1, random_state=42).reset_index(drop=True)
ds07['transaction_date'] = pd.date_range('2024-01-01', periods=n, freq='2min')
ds07['sender_id']   = RNG.integers(1, 5000, n)
ds07['receiver_id'] = RNG.integers(1, 3000, n)
save(ds07, "DS07_synthetic_rmt", "Données purement synthétiques → alerte RMT attendue")

# ─────────────────────────────────────────────────────────────────────────────
# DS08 — Label noise / Adversarial
# 10% des labels sont inversés (fraudes marquées légitimes, légitimes marquées
# fraudes). Simule une contamination de la vérité terrain : erreurs humaines,
# fraudes non déclarées, ou attaque adversariale sur les données d'entraînement.
# Attendu : MCC < 0.80, calibration dégradée, H-measure résistant
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DS08] Label noise / Adversarial...")
l8 = base_legit(N_LEGIT)
f8 = base_fraud(N_FRAUD)
ds08 = finalize(l8, f8)

# Inverser 10% des labels
n_noisy = int(0.10 * len(ds08))
noisy_idx = RNG.choice(len(ds08), n_noisy, replace=False)
ds08.loc[noisy_idx, 'is_fraud'] = 1 - ds08.loc[noisy_idx, 'is_fraud']
save(ds08, "DS08_label_noise", "10% de labels bruités — attaque adversariale")

# ─────────────────────────────────────────────────────────────────────────────
# Résumé
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*72}
  GÉNÉRATION TERMINÉE — {OUT}/
{'='*72}
  DS01_basic_fraud.csv         Baseline référence
  DS02_nocturnal_fraud.csv     Signal temporel unique
  DS03_extreme_imbalance.csv   0.2% fraude — Precision-Recall
  DS04_concept_drift.csv       Drift progressif dans le temps
  DS05_smurfing_benford.csv    Violation loi de Benford
  DS06_ring_network.csv        Réseau de 20 comptes complices
  DS07_synthetic_rmt.csv       Données purement synthétiques
  DS08_label_noise.csv         10% labels bruités
{'='*72}

Usage rapide :
  import pandas as pd
  from Fraud_Model_Validator_v3d import FraudModelValidatorV3

  df = pd.read_csv("DS05_smurfing_benford.csv", parse_dates=["transaction_date"])
  validator = FraudModelValidatorV3(model_name="Test DS05", cost_fn=5000, cost_fp=100)
  validator.benford_law_analysis(df, amount_col="transaction_amount")
""")
