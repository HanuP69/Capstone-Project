"""
Logistic Regression Meta-Learner for Groundedness Scoring Algorithm
===================================================================
Addresses Capstone Proposal Section 5.1:
"Deriving and justifying this weighting — and confirming it transfers across
 both the NITJ and secondary datasets without re-tuning — is the central
 technical and generalizability exercise of the project."

Replaces heuristic guesswork with a statistically principled Level-1 Stacking
Meta-Learner trained via Maximum Likelihood Estimation (MLE) over the
heterogeneous agent signals [F(c), P(c), D(c)].
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    log_loss, brier_score_loss
)

sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = "evaluation_results"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# 1. Load Datasets Across Institutions
# ============================================================================
primary_df = pd.read_csv("primary_audit_trail.csv")
secondary_df = pd.read_csv("secondary_audit_trail.csv")
mfaq_df = pd.read_csv("mfaq_agentic_audit_trail.csv")

print(f"Loaded Primary Dataset (NIT Jamshedpur): {len(primary_df)} claims")
print(f"Loaded Secondary Dataset (Public University): {len(secondary_df)} claims")
print(f"Loaded Benchmark Dataset (MFAQ): {len(mfaq_df)} claims")

X_primary = primary_df[["F", "P", "D"]].values
y_primary = primary_df["label"].values

# Binary ground truth: 1 for Grounded (true), 0 for Ungrounded (false/unverifiable)
y_primary_binary = (primary_df["label"] == "true").astype(int).values
# Hallucination ground truth: 1 for Hallucinated (false), 0 for Grounded/Neutral (true/unverifiable)
y_primary_hallucination = (primary_df["label"] == "false").astype(int).values

# ============================================================================
# 2. Fit Logistic Regression Meta-Learners
# ============================================================================
print("\n>>> Fitting Logistic Regression Meta-Learners via Maximum Likelihood Estimation...")

# 2A. Binary Logistic Regression (Grounded vs Ungrounded)
lr_binary = LogisticRegression(fit_intercept=True, C=1.0, random_state=42)
lr_binary.fit(X_primary, y_primary_binary)

raw_coefs = lr_binary.coef_[0]
intercept = lr_binary.intercept_[0]
pos_coefs = np.maximum(0, raw_coefs)
normalized_weights = pos_coefs / pos_coefs.sum()

w1_learned = round(float(normalized_weights[0]), 3)
w2_learned = round(float(normalized_weights[1]), 3)
w3_learned = round(float(normalized_weights[2]), 3)

print("--- Binary Meta-Learner (MLE Derivation) ---")
print(f"  Intercept beta_0: {intercept:.3f}")
print(f"  Raw Coefficients: beta_F={raw_coefs[0]:.3f}, beta_P={raw_coefs[1]:.3f}, beta_D={raw_coefs[2]:.3f}")
print(f"  Statistically Derived Normalized Weights:")
print(f"    w1 (Fact-Check Agent):        {w1_learned:.3f} ({w1_learned*100:.1f}%)")
print(f"    w2 (Policy-Compliance Agent): {w2_learned:.3f} ({w2_learned*100:.1f}%)")
print(f"    w3 (Grounding Agent):         {w3_learned:.3f} ({w3_learned*100:.1f}%)")

# 2B. 5-Fold Stratified Cross-Validation on Primary Dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_binary, X_primary, y_primary_binary, cv=skf, scoring="accuracy")
cv_auc = cross_val_score(lr_binary, X_primary, y_primary_binary, cv=skf, scoring="roc_auc")
print(f"  5-Fold Out-of-Fold CV Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
print(f"  5-Fold Out-of-Fold CV ROC-AUC:  {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")

# 2C. 3-Class Multinomial Meta-Learner (Direct PASS/FLAG/BLOCK decision)
lr_multi = LogisticRegression(max_iter=300, random_state=42)
lr_multi.fit(X_primary, y_primary)
multi_cv_scores = cross_val_score(lr_multi, X_primary, y_primary, cv=skf, scoring="accuracy")
print(f"\n--- 3-Class Multinomial Meta-Learner ---")
print(f"  5-Fold CV Accuracy (3 classes: True, False, Unverifiable): {multi_cv_scores.mean()*100:.1f}% (+/- {multi_cv_scores.std()*100:.1f}%)")
for c, coef in zip(lr_multi.classes_, lr_multi.coef_):
    print(f"  Log-odds for class '{c}': F={coef[0]:.2f}, P={coef[1]:.2f}, D={coef[2]:.2f}")


# ============================================================================
# 3. Systematic Comparison: 4 Weighting Methods Across 3 Datasets
# ============================================================================
print("\n>>> Systematically comparing 4 Weighting Methods across all 3 institutions/datasets...")

weighting_methods = [
    ("Method 1: Equal Weights (Baseline)", 0.333, 0.333, 0.334, "Equal a-priori weighting"),
    ("Method 2: Fixed Heuristic Weights", 0.400, 0.300, 0.300, "Empirical heuristic from proposal"),
    ("Method 3: Grid Search Tuned", 0.500, 0.300, 0.200, "Brute-force grid search on primary"),
    ("Method 4: Logistic Regression Meta-Learner (Ours)", w1_learned, w2_learned, w3_learned, "MLE-derived via Level-1 Stacking")
]

def evaluate_weight_configuration(w1, w2, w3, df, tau_pass=0.72, tau_flag=0.50):
    preds = []
    g_scores = []
    for _, r in df.iterrows():
        f, p, d = r["F"], r["P"], r["D"]
        if f >= 0.90:
            preds.append("true")
            g_scores.append(1.0)
            continue
        if f <= 0.10 or p <= 0.10:
            preds.append("false")
            g_scores.append(0.0)
            continue
        g = float(np.clip(w1 * f + w2 * p + w3 * d, 0.0, 1.0))
        g_scores.append(g)
        if g >= tau_pass:
            preds.append("true")
        elif g >= tau_flag:
            preds.append("unverifiable")
        else:
            preds.append("false")
    acc = accuracy_score(df["label"], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(df["label"], preds, average="macro", zero_division=0)
    
    # Hallucination Rate
    false_idx = [i for i, t in enumerate(df["label"]) if t == "false"]
    missed = sum(1 for i in false_idx if preds[i] != "false")
    hall_rate = (missed / len(false_idx)) * 100 if false_idx else 0.0

    return acc, f1, hall_rate

comparison_rows = []
for name, w1, w2, w3, desc in weighting_methods:
    p_acc, p_f1, p_hall = evaluate_weight_configuration(w1, w2, w3, primary_df)
    s_acc, s_f1, s_hall = evaluate_weight_configuration(w1, w2, w3, secondary_df)
    m_acc, m_f1, m_hall = evaluate_weight_configuration(w1, w2, w3, mfaq_df)
    
    avg_acc = (p_acc + s_acc + m_acc) / 3.0
    avg_f1 = (p_f1 + s_f1 + m_f1) / 3.0

    comparison_rows.append({
        "Weighting_Method": name,
        "w1_Fact": w1,
        "w2_Policy": w2,
        "w3_Grounding": w3,
        "Derivation_Basis": desc,
        "NITJ_Accuracy": round(p_acc * 100, 1),
        "PublicUniv_Accuracy": round(s_acc * 100, 1),
        "MFAQ_Accuracy": round(m_acc * 100, 1),
        "Cross_Dataset_Mean_Accuracy": round(avg_acc * 100, 1),
        "Cross_Dataset_Mean_Macro_F1": round(avg_f1, 3),
        "NITJ_Hallucination_Rate": round(p_hall, 1)
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n=== Weighting Method Comparison Table ===")
print(comparison_df[["Weighting_Method", "w1_Fact", "w2_Policy", "w3_Grounding", "NITJ_Accuracy", "PublicUniv_Accuracy", "MFAQ_Accuracy", "Cross_Dataset_Mean_Accuracy"]].to_string(index=False))

comparison_df.to_csv("meta_learner_weights_comparison.csv", index=False)
print("Saved meta_learner_weights_comparison.csv")


# ============================================================================
# 4. Generate Publication Plot: Weight Comparison & Odds Ratios
# ============================================================================
print(">>> Generating Publication Charts for Meta-Learner...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: Weight Breakdown Across Methods
methods_short = ["Equal", "Fixed Heuristic", "Grid Search", "LogReg Meta-Learner\n(MLE Stacking)"]
w_f = comparison_df["w1_Fact"].values
w_p = comparison_df["w2_Policy"].values
w_d = comparison_df["w3_Grounding"].values

x = np.arange(len(methods_short))
width = 0.55

p1 = ax1.bar(x, w_f, width, label="w1: Fact-Check Agent (Records)", color="#1a73e8")
p2 = ax1.bar(x, w_p, width, bottom=w_f, label="w2: Policy-Compliance Agent (Rules)", color="#34a853")
p3 = ax1.bar(x, w_d, width, bottom=w_f + w_p, label="w3: Grounding Agent (FAISS)", color="#fbbc04")

ax1.set_ylabel("Normalized Weight Contribution")
ax1.set_title("Groundedness Scoring Weight Derivation: Heuristic vs ML Meta-Learner")
ax1.set_xticks(x)
ax1.set_xticklabels(methods_short, fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.legend(loc="upper right", frameon=True)

# Plot 2: Cross-Dataset Mean Accuracy
bars_acc = ax2.bar(methods_short, comparison_df["Cross_Dataset_Mean_Accuracy"], color=["#9aa0a6", "#9aa0a6", "#9aa0a6", "#1a73e8"], width=0.5)
ax2.set_ylabel("Cross-Dataset Mean Accuracy (%)")
ax2.set_title("Cross-Dataset Generalization (NITJ + Public Univ + MFAQ)")
ax2.set_ylim(80, 100)

for b in bars_acc:
    h = b.get_height()
    ax2.annotate(f"{h:.1f}%",
                 xy=(b.get_x() + b.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, weight="bold")

plt.tight_layout()
meta_plot_path = os.path.join(FIG_DIR, "meta_learner_weight_comparison.png")
plt.savefig(meta_plot_path, dpi=300)
plt.close()
print(f"Saved {meta_plot_path}")


# ============================================================================
# 5. Export LaTeX Table for Academic Paper Insertion
# ============================================================================
latex_table = comparison_df[[
    "Weighting_Method", "w1_Fact", "w2_Policy", "w3_Grounding",
    "NITJ_Accuracy", "PublicUniv_Accuracy", "MFAQ_Accuracy", "Cross_Dataset_Mean_Accuracy"
]].to_latex(
    index=False,
    caption="Derivation of Groundedness Scoring Weights: Heuristic vs Logistic Regression Meta-Learner Across Institutions",
    label="tab:meta_learner_weights"
)

with open("meta_learner_results_table.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)
print("Saved meta_learner_results_table.tex")

print("\n=== Meta-Learner Formulation & Evaluation Finished Successfully ===")
