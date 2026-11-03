# SCBC

Implementation of **SCBC** (Subset-Conditioned Boundary Compensation) for multimodal learning
with missing modalities, on CMU-MOSEI and IEMOCAP-4.

SCBC keeps a modality-factorized violation memory (the debt book) of how often the masked view
of each training sample violates class margins beyond what the full-modality reference view
does. A subset-adaptive displacement head turns the debts of the observed subset into a
zero-sum logit correction, and the model is trained with cross-entropy on the corrected and
base logits plus a debt-weighted pairwise margin loss.

## Protocol

Fixed missing modalities: each run observes one fixed modality subset in every split, while the
reference view sees all modalities. The seven subsets are `l`, `v`, `a`, `lv`, `la`, `va` and
`lva` (l = text, a = audio, v = vision).

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. All runs use seed 42.

## Data

Place the pre-extracted feature pickles at

```text
data/mosei/CMUMOSEI_new.pkl          CMU-MOSEI, train/valid/test = 16326/1871/4659 segments
data/iemocap/IEMOCAP_4way_new.pkl    IEMOCAP-4, train/test = 3600/943 utterances
```

## Training

```bash
python train_mosei.py --subset lva --task binary --output runs/mosei/binary/lva
python train_mosei.py --subset lva --task seven --output runs/mosei/seven/lva
python train_iemocap4.py --subset lva --output runs/iemocap4/lva
```

On CMU-MOSEI the binary model gives Acc-2 and weighted F1 (neutral segments dropped) and the
seven-class model gives Acc-7. On IEMOCAP-4 accuracy, weighted F1 and macro F1 are logged.
Hyperparameters of every subset are in `configs/mosei.yaml` and `configs/iemocap4.yaml`.
Training logs go to `<output>/metrics.jsonl`.

## Layout

```text
train_mosei.py                  CMU-MOSEI entry: training and per-epoch evaluation
train_iemocap4.py               IEMOCAP-4 entry: training and per-epoch evaluation
configs/                        hyperparameters for CMU-MOSEI and IEMOCAP-4
scbc/
  core/
    debt_book.py                modality-factorized violation memory (DebtBook)
    corrector.py                subset-adaptive displacement head (DebtCorrector)
    objective.py                SCBC objective and debt-weighted pairwise margin loss
  backbones/
    subset_fusion.py            token-level subset fusion encoder (CMU-MOSEI)
    dialogue.py                 speaker-aware dialogue encoders and masked fusion head (IEMOCAP-4)
  models/
    mosei.py                    backbone + debt book / corrector / objective for CMU-MOSEI
    iemocap.py                  backbone + debt book / corrector / objective for IEMOCAP-4
  data/
    mosei.py                    CMU-MOSEI segments for the binary and seven-class tasks
    iemocap.py                  IEMOCAP-4 dialogues
  engine.py                     training epochs and evaluation
  metrics.py                    evaluation metrics
  utils.py                      seeding and config loading
  constants.py                  modalities, subsets and SCBC constants
```

## License

MIT, see [LICENSE](LICENSE).
