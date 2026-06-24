"""Project-wide constants.

Centralizes values that were previously hardcoded across modules so they have a
single, documented home.
"""

# Root directory for all generated models, caches, plots, and result JSONs.
RESULTS_DIR = "./results"

# Magnitude written into watermark-embedded samples (the "wonder filter").
OUT_OF_BOUND = 2000

# Seeds for reproducible trigger-set sampling. Deliberately fixed and decoupled
# from the experiment-wide --seed so that the trigger set is identical across
# runs and folds, which watermark verification relies on.
TRIGGERSET_SEED = 2036
VERIFICATION_SEED = 42

# Default (true, null) trigger-set sizes.
DEFAULT_TRIGGERSET_SIZE = (1600, 1600)

# Watermark verifier identity strings, signed and hashed into each watermark.
# Externalized so they can be changed without touching watermark logic.
OWNER_IDENTITY = "Abdelaziz->AHMED a.k.a OWNER<-Fathi @ Feb 15, 2025"
NON_OWNER_IDENTITY = "Abdelaziz->NOT OWNER<-Fathi @ Feb 15, 2025"
ATTACKER_IDENTITY = "Abdelaziz->ATTACKER<-Fathi @ Feb 15, 2025"
