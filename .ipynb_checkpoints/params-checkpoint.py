"""
params.py
=========
Central configuration for the full Target Classification training pipeline.
Covers both the Target-Type branch (main-type.ipynb) and the
Intent branch (main-intent.ipynb).

Edit values here; every notebook and script that does ``import params``
will pick up the changes automatically.
"""

import os

# ===========================================================================
# SHARED
# ===========================================================================

BASE_DIR = os.getcwd()
INPUT_PATH = os.path.join(BASE_DIR, "Input", "trajectory.csv")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

# Sliding-window shared parameter
M  = 1000    # total time-steps per trajectory
W  = 50      # sliding-window size

# ===========================================================================
# TARGET-TYPE BRANCH  (main-type.ipynb)
# ===========================================================================

# ── Feature dimensions ──────────────────────────────────────────────────────
K1          = 9       # number of raw features per time-step

# ── Split ratios ────────────────────────────────────────────────────────────
TYPE_TEST_SIZE  = 0.2     # fraction held out for testing
TYPE_VAL_SIZE   = 0.125   # fraction of remaining train set used for validation

# ── Model architecture ──────────────────────────────────────────────────────
NUM_CLASS   = 8       # number of target-type classes

# ── Model save paths ────────────────────────────────────────────────────────
SCALER_STAT_PATH      = os.path.join(TRAINED_MODELS_DIR, "scaler_model_stat.pkl")
SCALER_TRANSFORM_PATH = os.path.join(TRAINED_MODELS_DIR, "scaler_model_transform.pkl")
OHE_PATH              = os.path.join(TRAINED_MODELS_DIR, "onehot_encoder_model.pkl")
FFNN_MODEL_PATH       = os.path.join(TRAINED_MODELS_DIR, "ffnn_model.keras")
LSTM_MODEL_PATH       = os.path.join(TRAINED_MODELS_DIR, "lstm_model.keras")
SVM_MODEL_PATH        = os.path.join(TRAINED_MODELS_DIR, "svm_model.pkl")

# ── Output paths ────────────────────────────────────────────────────────────
TYPE_OUTPUT_DIR       = os.path.join(BASE_DIR, "output", "target_type_belief_distribution")
TYPE_TRAIN_OUTPUT_CSV = os.path.join(TYPE_OUTPUT_DIR, "train_belief_distribution.csv")
TYPE_TEST_OUTPUT_CSV  = os.path.join(TYPE_OUTPUT_DIR, "test_belief_distribution.csv")

# ── Accuracy registry (written back by main-type.ipynb after training) ──────
# fmt: off
FFNN_ACCURACY: float = 0.808062    
LSTM_ACCURACY: float = 0.747063  
SVM_ACCURACY:  float = 0.7595  
# fmt: on

# ===========================================================================
# INTENT BRANCH  (main-intent.ipynb)
# ===========================================================================

# ── Feature dimensions ──────────────────────────────────────────────────────
K2          = 4       # number of raw features per time-step (intent branch)

# ── Split ratios ────────────────────────────────────────────────────────────
INTENT_TEST_SIZE = 0.2    # fraction held out for testing (no validation split)

# ── Temporal fusion ─────────────────────────────────────────────────────────
INTENT_BATCH_SIZE = 20    # batch size for temporal fusion
OPTIMAL_X: float  = 0.7356230806

# ── Model save path ─────────────────────────────────────────────────────────
DT_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "decision_tree_model.pkl")

# ── Output paths ────────────────────────────────────────────────────────────
INTENT_OUTPUT_DIR      = os.path.join(BASE_DIR, "output", "target_intent_belief_distribution")
INTENT_TEST_OUTPUT_CSV = os.path.join(INTENT_OUTPUT_DIR, "test_belief_distribution.csv")

# ── Accuracy registry (written back by main-intent.ipynb after training) ────
# fmt: off
DT_ACCURACY:  float = 0.9282500000
# fmt: on

# ===========================================================================
# INFERENCE / DEMO  (demo.ipynb)
# ===========================================================================

# ── Label mappings ──────────────────────────────────────────────────────────
TARGET_TYPE_LABELS = {
    0: "Bomber",
    1: "CivilianCommercialAirliner",
    2: "CivilianHelicopter",
    3: "CruiseMissile",
    4: "MilitaryHelicopter",
    5: "MultiroleFighter",
    6: "SmallCivilian",
    7: "SmallUnmanned",
    8: r"\Theta_1",
    9: r"\Theta_2",
    10: r"\Omega",
}

INTENT_LABELS = {
    0: "Hostile",
    1: "Non Hostile",
    2: "Suspicious",
}

INTENT_TEMPORAL_LABELS = {
    0: "Hostile",
    1: "Non Hostile",
    2: "Suspicious",
    3: r"\Omega",
}

# ── Output table column headers ─────────────────────────────────────────────
TIME_STEPS = [f"t_{i}" for i in range(1, 21)]

TARGET_TYPE_COLS = [
    "", "Bomber", "CivilianCommercialAirliner", "CivilianHelicopter",
    "zCruiseMissile", "MilitaryHelicopter", "MultiroleFighter",
    "SmallCivilian", "SmallUmanned",
    r"$\Theta_1$", r"$\Theta_2$", r"$\Omega$",
    "True Label", "Predicted Label",
]

INTENT_DT_COLS    = ["", "Hostile", "Non Hostile", "Suspicious", "True Label", "Predicted Label"]
INTENT_FUSED_COLS = ["", "Hostile", "Non Hostile", "Suspicious", "Ω", "True Label", "Predicted Label"]

# ── Demo output directory (filename is built dynamically in the notebook) ───
DEMO_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "demo")
