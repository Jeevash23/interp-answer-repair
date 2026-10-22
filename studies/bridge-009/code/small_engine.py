"""GPT-2 Small constructor with the validated, generic causal-prefix algorithm.

Engine003's non-Medium branch loads Pythia. Deliberately bypass that constructor
and suffix dispatcher; preserve the original GPT-2 model configuration instead.
Importing this module does not load a checkpoint or perform model inference.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "studies/bridge-006/code")]
from cpu_engine import CPUEngine
from assay import CausalEngine
from transport_bridge.engine import Engine

FullSequenceEngine = CausalEngine


class SmallCPUEngine(CPUEngine):
    def __init__(self):
        config = json.loads((ROOT / "configs/study.json").read_text())
        CausalEngine.__init__(self, "gpt2-small", config, "cpu")
        assert self.site == 7 and self.layers == list(range(7, 12))
        assert self.heads == 12 and self.spec["width"] == 768
        assert self.source == [(0, 1), (3, 0), (5, 5), (6, 9)]
        self.model_name = "gpt2-small"

    suffix = Engine.suffix
