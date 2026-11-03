"""Constants shared by the data pipelines, the SCBC core and both benchmarks."""

MODALITIES = ("text", "audio", "vision")
FIELDS = {"text": "text_features", "audio": "audio_features", "vision": "visual_features"}
LETTERS = {"l": "text", "a": "audio", "v": "vision"}
SUBSETS = ("l", "v", "a", "lv", "la", "va", "lva")

TAU_M = 1.0              # target margin tau_m
MU = 0.9                 # EMA rate of the excess-violation state V
MU_N = 0.95              # EMA rate of the class support N
NU = 0.5                 # bandwidth nu of the class-pair indicator, also the shrinkage constant nu_s
OMEGA = 1.0              # concentration sharpening omega
BETA_MAX = 1.0           # bound of the per-class step modulator
EPS = 1.0e-6
EPS_W = 1.0e-6           # Laplace smoothing of the pairwise weights
CORRECTOR_HIDDEN_DIM = 256
CORRECTOR_DROPOUT = 0.1
LAMBDA_PAIR = 0.5


def observed_mask(subset):
    """[M] list with 1.0 for every modality observed in the subset string (e.g. "la")."""
    observed = {LETTERS[ch] for ch in subset}
    return [1.0 if m in observed else 0.0 for m in MODALITIES]
