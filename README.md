# 🧪 Fraud Datasets — Banc de test pour FraudModelValidatorV3

> **8 datasets synthétiques progressifs**, du cas trivial au piège adversarial,  
> conçus pour exercer chaque section du framework de validation.

---

## 📖 Comment lire ce README

Ce document est organisé comme un **cours progressif** :

1. D'abord, comprendre *pourquoi* ces datasets existent et comment ils sont construits
2. Ensuite, découvrir chaque scénario avec son intention pédagogique
3. Enfin, savoir comment les utiliser concrètement avec le framework

Vous pouvez lire linéairement, ou aller directement à la [table des scénarios](#-les-8-scénarios-de-a-à-z) si vous êtes déjà familier du contexte.

---

## 🗂️ Fichiers du repo

```
datasets/
│
├── generate_datasets.py          ← Script de génération (reproductible, seed=42)
├── run_all_tests.py              ← Test runner : lance les 8 validations d'un coup
│
├── DS01_basic_fraud.csv          ← Baseline — séparation parfaite
├── DS02_nocturnal_fraud.csv      ← Signal temporel unique
├── DS03_extreme_imbalance.csv    ← 0.2% de fraude
├── DS04_concept_drift.csv        ← Profil de fraude qui évolue dans le temps
├── DS05_smurfing_benford.csv     ← Violation de la loi de Benford
├── DS06_ring_network.csv         ← Réseau de 20 comptes complices
├── DS07_synthetic_rmt.csv        ← Données purement synthétiques (piège RMT)
└── DS08_label_noise.csv          ← 10% de labels bruités (adversarial)
```

---

## 🧱 Structure commune de tous les datasets

Chaque fichier CSV contient exactement les mêmes colonnes, conformes au schéma attendu par `FraudModelValidatorV3` :

| Colonne | Type | Description | Plage typique |
|---|---|---|---|
| `transaction_amount` | float | Montant de la transaction en € | 1 – 50 000 |
| `transaction_velocity_1h` | int | Nombre de tx de ce client dans la dernière heure | 0 – 20 |
| `merchant_risk_score` | float | Score de risque du commerçant (calculé en amont) | 0.0 – 1.0 |
| `customer_age_days` | float | Ancienneté du compte client en jours | 1 – 3 650 |
| `is_international` | 0/1 | Transaction transfrontalière | 0 ou 1 |
| `hour_of_day` | int | Heure locale de la transaction | 0 – 23 |
| `device_risk_score` | float | Score de risque du terminal/device | 0.0 – 1.0 |
| `is_fraud` | 0/1 | **Label vérité terrain** | 0 ou 1 |
| `transaction_date` | datetime | Horodatage complet | 2024-01-01 → |
| `sender_id` | int | Identifiant de l'émetteur | 1 – 5 000 |
| `receiver_id` | int | Identifiant du destinataire | 1 – 3 000 |

> **Note :** `transaction_date`, `sender_id` et `receiver_id` ne sont pas utilisés comme features par le modèle GBM, mais sont requis par plusieurs modules du framework : `walk_forward_backtest` (date), `graph_anomaly_detection` (sender/receiver), `ultra_scientific_detection` (date pour les séries temporelles).

---

## 🧠 Principe de construction

### Profil d'une transaction légitime (base)

```
transaction_amount      ~ LogNormal(μ=4.0, σ=1.5)   → montants variés, pic autour de 55€
transaction_velocity_1h ~ Poisson(λ=2)               → 0-5 tx/heure en moyenne
merchant_risk_score     ~ Beta(α=2, β=5)             → distribution vers les faibles risques
customer_age_days       ~ Exponentielle(λ=1/365)     → beaucoup de clients récents
is_international        ~ Bernoulli(p=0.05)          → 5% de tx internationales
hour_of_day             ~ Uniforme[8, 19]            → heures de bureau
device_risk_score       ~ Beta(α=1, β=5)             → terminaux généralement sains
```

### Profil d'une transaction frauduleuse (base)

```
transaction_amount      ~ LogNormal(μ=6.5, σ=1.0)   → montants plus élevés (pic ~665€)
transaction_velocity_1h ~ Poisson(λ=10)              → rafale de tx (fraude en série)
merchant_risk_score     ~ Beta(α=6, β=2)             → commerçants à risque élevé
customer_age_days       ~ Exponentielle(λ=1/20)      → comptes très récents (mule)
is_international        ~ Bernoulli(p=0.50)          → 50% de tx internationales
hour_of_day             ~ {0,1,2,3,4,5,22,23}       → nuit profonde
device_risk_score       ~ Beta(α=5, β=1)             → terminaux suspects
```

Chaque dataset **modifie ce profil de base** pour créer le scénario voulu. C'est ce que les sections suivantes détaillent.

---

## 🎯 Les 8 scénarios — de A à Z

---

### DS01 — Fraude basique `baseline`

**Fichier :** `DS01_basic_fraud.csv`  
**Taux de fraude :** 2.50% (250 fraudes / 10 000 tx)

#### Ce que ce dataset représente

Le cas idéal, presque irréaliste : **toutes les features discriminent fortement** entre légitimes et fraudes. Les deux populations sont très séparées dans l'espace des features. C'est le dataset de référence — si votre modèle échoue ici, il y a un problème de pipeline.

#### Ce qu'il teste dans le framework

- `calculate_performance_metrics` → doit retourner AUC > 0.97, KS > 0.90
- `calibration_analysis` → Brier Score très faible attendu
- `benford_law_analysis` → montants légitimes suivent Benford → conforme

#### Résultat observé lors des tests

```
AUC-ROC : 1.0000  |  KS : 1.0000  |  F1 : 1.0000  |  Benford : Conforme
```

#### ⚠️ Ce que ce résultat parfait doit vous apprendre

Un AUC de 1.0000 sur données synthétiques aussi séparées est **normal et attendu**. Sur données réelles, tout AUC > 0.99 doit déclencher une investigation de data leakage (cf. DS07 et DS08 pour voir ce que le framework détecte quand ça va mal).

---

### DS02 — Fraude nocturne `signal unique`

**Fichier :** `DS02_nocturnal_fraud.csv`  
**Taux de fraude :** 2.50%

#### Ce que ce dataset représente

Une situation plus subtile : **les fraudes ressemblent aux transactions légitimes sur tous les plans** (même montants, même vélocité, même merchants) — à une exception près : elles surviennent **uniquement entre 00h et 05h**.

```python
# Override appliqué aux fraudes dans DS02 :
f2['transaction_amount']      = LogNormal(4.0, 1.5)   # identique aux légitimes
f2['transaction_velocity_1h'] = Poisson(2)             # identique
f2['merchant_risk_score']     = Beta(2, 5)             # identique
f2['hour_of_day']             = Uniforme[0, 4]         # ← SEULE différence
```

#### Ce qu'il teste dans le framework

- **La capacité du modèle à capturer un signal unique** : un GBM bien configuré doit isoler `hour_of_day` comme feature dominante
- `ultra_scientific_detection` → la série temporelle des scores doit montrer une périodicité journalière → BOCD peut détecter le rythme
- `optimize_threshold` → utile car le signal est propre mais unique

#### À observer

En pratique sur des données réelles, la fraude nocturne pure est rare. Ce dataset sert à vérifier que votre pipeline de features ne perd pas l'information temporelle (certains pipelines scalent ou transforment l'heure de façon à perdre son caractère cyclique).

---

### DS03 — Déséquilibre extrême `0.2% de fraude`

**Fichier :** `DS03_extreme_imbalance.csv`  
**Taux de fraude :** 0.20% (20 fraudes / 10 000 tx)

#### Ce que ce dataset représente

La fraude bancaire réelle est souvent à moins de 0.5%. Ce dataset pousse cela à l'extrême : **20 cas positifs seulement** dans 10 000 transactions. Le profil des fraudes est identique à DS01 (très discriminant), mais la rareté extrême crée des défis spécifiques.

#### Ce qu'il teste dans le framework

- **Precision-Recall vs ROC** : l'AUC-ROC peut rester élevée même avec un modèle qui ne détecte rien (car les vrais négatifs sont 99.8% des cas). La courbe Precision-Recall révèle la vraie performance.
- `calculate_performance_metrics` → comparer AUC (optimiste) vs Average Precision (réaliste)
- `optimize_threshold` → le seuil par défaut de 0.5 est inutilisable — utiliser la fonction de coût

#### Règle clé à comprendre

```
Sur un dataset à 0.2% de fraude, un modèle qui prédit TOUJOURS "légitime" obtient :
  Accuracy    = 99.8%    (trompeusement parfaite)
  Recall      = 0.0%     (catastrophique)
  F1          = 0.0%     (révélateur)
  MCC         = 0.0%     (révélateur)

→ Toujours reporter MCC et F1, jamais seulement l'Accuracy.
```

---

### DS04 — Concept drift `fraude évolutive`

**Fichier :** `DS04_concept_drift.csv`  
**Taux de fraude :** 2.50%

#### Ce que ce dataset représente

Le profil de fraude **change radicalement à mi-parcours** du dataset :

| Période | Profil fraude | Signal dominant |
|---|---|---|
| Janvier → Juin (phase 1) | Montants élevés, nuit, vélocité forte | `transaction_amount` + `hour_of_day` |
| Juillet → Décembre (phase 2) | Petits montants (~30€), nuit, vélocité modérée | `hour_of_day` seul |

```python
# Phase 2 — fraude migre vers petits montants discrets
f4b['transaction_amount']  = LogNormal(3.5, 0.5)   # petits montants ~30€
f4b['hour_of_day']         = Uniforme[0, 3]         # toujours nocturne
f4b['merchant_risk_score'] = Beta(2, 4)             # moins marqué qu'avant
```

#### Ce qu'il teste dans le framework

C'est le dataset idéal pour tester les **analyses temporelles** du framework :

- `walk_forward_backtest` → l'AUC doit décroître sur les derniers folds (le modèle entraîné sur phase 1 rate la phase 2)
- `ultra_scientific_detection` → BOCD (Bayesian Online Changepoint Detection) doit identifier la rupture à mi-parcours
- `stability_analysis` → PSI élevé entre première et deuxième moitié

#### Comment le mettre en évidence

```python
validator.walk_forward_backtest(
    df, model=votre_modele,
    feature_cols=FEATURE_COLS,
    target_col="is_fraud",
    date_col="transaction_date",
    n_splits=6    # 6 folds pour voir la dégradation progressive
)
```

Vous devriez voir l'AUC stable puis chuter après le 3ème fold.

---

### DS05 — Smurfing / Benford `violation KYC`

**Fichier :** `DS05_smurfing_benford.csv`  
**Taux de fraude :** 2.50%

#### Ce que ce dataset représente

Le **smurfing** (ou "structuring") est une technique de blanchiment qui consiste à fractionner des montants importants en transactions juste **en dessous des seuils de déclaration réglementaires** :

```
Seuil KYC / déclaration automatique : 10 000€
  → Fraudeur envoie 9 997.43€, 9 998.12€, 9 995.67€...

Seuil de vigilance renforcée : 5 000€
  → 4 998.55€, 4 997.23€...

Seuil d'alerte commerçant : 3 000€
  → 2 997.84€, 2 999.01€...
```

#### Comment la Loi de Benford le détecte

La loi de Benford prédit que dans des données financières naturelles, les premiers chiffres significatifs suivent une distribution logarithmique :

```
Chiffre 1 : 30.1% des montants commencent par 1
Chiffre 2 : 17.6%
Chiffre 3 : 12.5%
...
Chiffre 9 :  4.6%
```

Avec du smurfing autour de 9 999€, 4 999€, 2 999€ :
- **Le chiffre 9** est massivement surreprésenté (9 999, 9 998…)
- **Le chiffre 4** est surreprésenté (4 999, 4 998…)
- **Le chiffre 2** est surreprésenté (2 999, 2 998…)

Le χ² test détecte cette déviation statistiquement significative.

#### Résultat observé lors des tests

```
MAD        : 0.00348
χ² p-value : 0.00001   ← Anomalie hautement significative
Conformité : ⚠️ Anomalie détectée
```

#### Subtilité importante

Le modèle GBM obtient quand même un AUC de 1.0000 sur ce dataset, car les autres features (vélocité, heure…) discriminent encore. **Benford détecte ce que le modèle ne remonte pas** : la manipulation des montants en elle-même, indépendamment de l'issue de la classification.

---

### DS06 — Réseau organisé `graph anomaly`

**Fichier :** `DS06_ring_network.csv`  
**Taux de fraude :** 2.50%

#### Ce que ce dataset représente

Un **anneau de blanchiment** : 20 comptes complices (IDs 90001 à 90020) s'envoient mutuellement de l'argent en boucle fermée. Toutes les transactions frauduleuses impliquent exclusivement ces comptes entre eux.

```
90001 → 90007 → 90015 → 90003 → 90018 → 90001 → ...
   ↑                                          ↓
90012 ← 90005 ← 90019 ← 90009 ← 90011 ← 90002
```

En termes de graphe de transactions, ces 20 nœuds forment une **clique très dense** avec des degrés entrants/sortants anormalement élevés, alors que les 5 000 autres comptes n'ont généralement que 1-3 connexions.

#### Ce qu'il teste dans le framework

```python
validator.graph_anomaly_detection(
    df,
    source_col="sender_id",
    target_col="receiver_id",
    amount_col="transaction_amount",
    fraud_col="is_fraud"
)
```

Le module construit un graphe `networkx.DiGraph` et calcule pour chaque nœud :
- **Degré entrant / sortant** (hubs suspects)
- **Betweenness centrality** (nœuds "pont")
- **Détection de communautés** (clusters denses)

Les comptes 90001–90020 doivent apparaître dans les top anomalies.

#### Pourquoi c'est important

Un modèle de scoring par transaction ne voit qu'une transaction à la fois. Il peut très bien scorer chaque transaction de l'anneau comme "normale" (montants raisonnables, heures correctes) mais **rater le pattern global**. L'analyse de graphe est complémentaire, pas redondante.

---

### DS07 — Données synthétiques `piège RMT`

**Fichier :** `DS07_synthetic_rmt.csv`  
**Taux de fraude :** 2.50%

#### Ce que ce dataset représente

Toutes les features sont générées **indépendamment** par des lois uniformes, sans aucune structure sous-jacente :

```python
# DS07 — aucune corrélation naturelle
transaction_amount      = Uniforme(10, 10 000)
transaction_velocity_1h = Uniforme_entier(0, 15)
merchant_risk_score     = Uniforme(0, 1)
customer_age_days       = Uniforme(1, 3 650)
is_international        = Uniforme_entier(0, 1)
hour_of_day             = Uniforme_entier(0, 24)
device_risk_score       = Uniforme(0, 1)

# Labels assignés au hasard (2.5% de fraude aléatoire)
```

Il n'y a **aucune relation** entre les features et le label. Le modèle ne peut rien apprendre.

#### Ce qu'il teste dans le framework

C'est le dataset qui doit déclencher les alertes **ultra-scientifiques** :

**Random Matrix Theory (RMT)**

Sur des données réelles, la matrice de corrélation entre features contient des corrélations "naturelles" (ex : heure et vélocité sont liées). Ces corrélations produisent une distribution des valeurs propres qui s'écarte de façon prévisible de la distribution de Marchenko-Pastur (distribution des valeurs propres d'une matrice aléatoire pure).

Sur DS07, **toutes les corrélations sont purement aléatoires** → la matrice de corrélation ressemble exactement à celle d'une matrice aléatoire → le test RMT doit détecter l'anomalie.

**Benford's Law**

Des montants uniformes entre 10 et 10 000€ violent massivement Benford (tous les chiffres initiaux auraient la même fréquence ~11%, alors que Benford prédit 30% pour le chiffre 1).

#### Résultats observés lors des tests

```
AUC-ROC    : 0.4621   ← Pire qu'aléatoire (0.50 attendu, bruit de variance)
GINI       : -0.076   ← Négatif : le modèle est inversé
MCC        : -0.009   ← Quasi-nul
Benford    : ⚠️ Non-conforme (MAD=0.058, χ² p≈0)
```

Trois alertes automatiques levées. C'est exactement ce que le framework doit faire : **refuser de valider un modèle sans signal**.

---

### DS08 — Label noise `attaque adversariale`

**Fichier :** `DS08_label_noise.csv`  
**Taux de fraude apparent :** ~12% (après bruitage)

#### Ce que ce dataset représente

Le profil de fraude est identique à DS01 (très discriminant), mais **10% des labels sont inversés** après génération :

```python
# Inverser 10% des labels au hasard
noisy_idx = RNG.choice(len(ds08), 1000, replace=False)
ds08.loc[noisy_idx, 'is_fraud'] = 1 - ds08.loc[noisy_idx, 'is_fraud']
```

Concrètement :
- ~75 fraudes réelles sont re-labelisées "légitimes" → le modèle ne les voit jamais
- ~925 transactions légitimes sont re-labelisées "fraudes" → le modèle apprend de faux positifs

Cela simule plusieurs situations réelles : erreurs humaines dans la vérité terrain, fraudes non déclarées (undeclared fraud), attaque adversariale sur les données d'entraînement, ou mauvaise jointure entre systèmes de décision et base de cas avérés.

#### Ce qu'il teste dans le framework

Le bruitage de labels affecte différemment les métriques :

| Métrique | Sensibilité au label noise | Raison |
|---|---|---|
| AUC-ROC | Moyenne | Mesure le ranking, partiellement robuste |
| MCC | Haute | Pénalise fortement FP et FN simultanément |
| F1 | Haute | Chute si précision ET rappel dégradés |
| Brier Score | Haute | Pénalise les probabilités mal calibrées |
| H-Measure | Faible | Plus robuste aux déséquilibres |

#### Résultats observés lors des tests

```
AUC-ROC    : 0.6033   ← Dégradé (vs 1.0000 sur DS01 identique sans bruit)
KS         : 0.1977   ← Très dégradé
MCC        : 0.3644   ← Alerte 🟡
F1         : 0.2936   ← Chute sévère
Brier      : 0.0918   ← ×900 vs DS01
```

#### Comment l'interpréter en pratique

Si vous observez ce pattern (AUC modéré, MCC/F1 très dégradés, Brier élevé) sur votre modèle de production, les causes à investiguer en priorité sont :
1. La qualité de votre vérité terrain (processus de labellisation)
2. Le délai de remontée des cas confirmés (fraud lag)
3. Une possible contamination de votre pipeline de données

---

## 🚀 Utilisation rapide

### Installation des dépendances

```bash
pip install pandas numpy scikit-learn scipy statsmodels shap networkx antropy
```

### Générer les datasets (si vous voulez les recréer)

```bash
python generate_datasets.py
```

Les 8 fichiers CSV sont créés dans le dossier courant. La génération est reproductible (`seed=42`).

### Lancer tous les tests d'un coup

```bash
# Validation complète (toutes les sections sauf ultra-scientific)
python run_all_tests.py

# Avec ultra-scientific (plus lent, ~10 min pour les 8 datasets)
python run_all_tests.py --ultra

# Avec walk-forward backtesting (recommandé pour DS04)
python run_all_tests.py --walk-forward

# Un seul dataset
python run_all_tests.py --ds DS05_smurfing_benford

# Désactiver des sections pour aller plus vite
python run_all_tests.py --no-graph --no-benford
```

### Utiliser un dataset directement dans votre code

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Fraud_Model_Validator_v3d import FraudModelValidatorV3

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv("DS05_smurfing_benford.csv", parse_dates=["transaction_date"])

FEATURE_COLS = [
    "transaction_amount", "transaction_velocity_1h", "merchant_risk_score",
    "customer_age_days", "is_international", "hour_of_day", "device_risk_score",
]

# ── Modèle ────────────────────────────────────────────────────────────────────
X, y = df[FEATURE_COLS], df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                     stratify=y, random_state=42)
scaler = StandardScaler()
model  = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(scaler.fit_transform(X_train), y_train)
y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]

# ── Validation ────────────────────────────────────────────────────────────────
validator = FraudModelValidatorV3(model_name="Mon modèle", cost_fn=5000, cost_fp=100)

# Performance générale
metrics = validator.calculate_performance_metrics(y_test.values, y_proba)

# Section spécifique à DS05 : Benford
benford = validator.benford_law_analysis(df, amount_col="transaction_amount")
# → MAD élevé + χ² p < 0.05 → anomalie Benford détectée ✅

# Rapport complet
validator.generate_html_report("rapport_DS05.html")
```

---

## 📊 Tableau de résultats de référence

Ces résultats ont été obtenus lors de l'exécution de `run_all_tests.py` avec un GBM standard (`n_estimators=100, max_depth=4`). Ils servent de **ligne de base** pour détecter des régressions dans le framework.

| Dataset | Fraude% | AUC | KS | MCC | Brier | Benford | Alertes |
|---|---:|---:|---:|---:|---:|---|---|
| DS01 Basique | 2.50% | 1.0000 | 1.000 | 1.000 | 0.000 | ✅ | — |
| DS02 Nocturne | 2.50% | 1.0000 | 1.000 | 1.000 | 0.000 | ✅ | — |
| DS03 0.2% fraude | 0.20% | 1.0000 | 1.000 | 1.000 | 0.000 | ✅ | — |
| DS04 Drift | 2.50% | 1.0000 | 1.000 | 1.000 | 0.000 | ✅ | Walk-fwd |
| DS05 Smurfing | 2.50% | 1.0000 | 1.000 | 1.000 | 0.000 | ⚠️ χ²≈0 | Benford |
| DS06 Réseau | 2.50% | 0.9996 | 0.997 | 0.975 | 0.001 | ✅ | — |
| **DS07 Synthétique** | 2.50% | **0.462** | 0.026 | -0.009 | 0.027 | 🔴 | AUC + Benford |
| **DS08 Label noise** | 12.0% | **0.603** | 0.198 | 0.364 | 0.092 | ✅ | MCC + Brier |

### Lecture du tableau

**DS01–DS04 :** AUC = 1.000. Ce n'est pas du data leakage — c'est simplement la conséquence d'une séparation très nette dans les données synthétiques (voir le [README principal](../README.md) pour la discussion sur ce sujet). En conditions réelles, les AUC seront bien inférieurs.

**DS05 :** Le modèle classe parfaitement les fraudes, mais Benford alerte quand même sur les montants. Ces deux informations sont **complémentaires**, pas redondantes : l'une concerne la qualité du modèle, l'autre la qualité des données.

**DS07 :** Trois alertes simultanées sur un dataset sans signal. C'est la réponse attendue : le framework doit *refuser* de valider un modèle qui n'a rien appris.

**DS08 :** La dégradation du MCC (1.000 → 0.364) et du Brier (0.000 → 0.092) révèle la contamination des labels, même si l'AUC reste modérément acceptable (0.603). Illustration parfaite de pourquoi il faut rapporter plusieurs métriques.

---

## 🗺️ Guide : quel dataset pour quel objectif ?

| Objectif | Dataset recommandé | Section du framework ciblée |
|---|---|---|
| Vérifier que mon pipeline fonctionne | DS01 | `calculate_performance_metrics` |
| Tester la calibration | DS01, DS08 | `calibration_analysis` |
| Tester l'optimisation de seuil | DS02, DS03 | `optimize_threshold` |
| Détecter du concept drift | DS04 | `walk_forward_backtest`, BOCD |
| Détecter du smurfing/structuring | DS05 | `benford_law_analysis` |
| Détecter un réseau de fraude | DS06 | `graph_anomaly_detection` |
| Tester les alertes ultra-scientifiques | DS07 | `ultra_scientific_detection` (RMT) |
| Évaluer la robustesse au bruit | DS08 | `calculate_performance_metrics` (MCC) |
| Scénario de stress test complet | DS07 + DS08 | `stress_test` |
| Benchmark complet framework | Tous les 8 | `run_all_tests.py` |

---

## 🔧 Personnalisation

### Modifier les paramètres de génération

```python
# Dans generate_datasets.py, ajuster en tête de fichier :
N_LEGIT = 9750   # Transactions légitimes
N_FRAUD = 250    # Transactions frauduleuses (taux = N_FRAUD / (N_LEGIT + N_FRAUD))

# Changer la seed pour obtenir un jeu différent
RNG = np.random.default_rng(1234)   # seed=42 par défaut
```

### Créer votre propre scénario

```python
# Exemple : fraude par carte volée (montants élevés, device inconnu, heure variable)
l_custom = base_legit(9750)
f_custom = base_fraud(250)

# Override : device toujours inconnu, montant très élevé
f_custom['device_risk_score']    = np.ones(250) * 0.99   # device quasi-certain risqué
f_custom['transaction_amount']   = RNG.uniform(500, 5000, 250)  # montants élevés
f_custom['transaction_velocity_1h'] = RNG.poisson(1, 250)  # une seule tx (discret)

ds_custom = finalize(l_custom, f_custom)
save(ds_custom, "DS09_stolen_card", "Fraude par carte volée")
```

### Adapter les coûts au contexte métier

```python
# Banque retail (coûts standards)
validator = FraudModelValidatorV3(cost_fn=5000, cost_fp=100)

# E-commerce (coûts plus faibles)
validator = FraudModelValidatorV3(cost_fn=200, cost_fp=5)

# Virement B2B (coûts très élevés)
validator = FraudModelValidatorV3(cost_fn=50000, cost_fp=500)
```

---

## ❓ FAQ

**Q : Pourquoi utiliser des données synthétiques plutôt que des données réelles ?**  
R : Les données réelles de fraude sont confidentielles, non partageables, et très dépendantes du contexte métier. Les datasets synthétiques permettent de vérifier que le *framework lui-même* fonctionne correctement, indépendamment du modèle et des données. C'est un test de la *validation*, pas du *modèle*.

**Q : Les AUC à 1.0000 sur DS01-DS04 sont-ils réalistes ?**  
R : Non, et c'est voulu. Ces datasets sont délibérément simples pour que le modèle les résolve facilement. L'intérêt pédagogique est ailleurs : observer ce que chaque *section du framework* remonte, indépendamment de la performance du modèle. DS07 et DS08 sont les vrais tests de robustesse.

**Q : Puis-je utiliser ces datasets pour entraîner un modèle de production ?**  
R : Non. Ces données sont entièrement synthétiques et ne reflètent pas les distributions réelles de la fraude dans votre contexte. Elles sont exclusivement destinées à tester le framework de validation.

**Q : Comment reproduire exactement les mêmes datasets ?**  
R : La seed est fixée (`seed=42` dans `generate_datasets.py`). Avec les mêmes dépendances (numpy version identique), les fichiers générés seront bit-pour-bit identiques.

**Q : Le runner détecte `RMT p=0.000` sur DS01–DS06, est-ce une erreur ?**  
R : C'est une limitation de la version sans `--ultra` : la colonne `rmt_pvalue` vaut 0 par défaut quand l'ultra-detection n'est pas activée. Lancer avec `python run_all_tests.py` (ultra activé par défaut) pour obtenir les vraies valeurs RMT.

---

*Datasets générés avec `seed=42` — FraudModelValidatorV3 | SR 11-7 · ECB TRIM · AMLD6 · Bâle III*
