## 2024-05-10 - [Optimize MultiheadAttention with need_weights=False]
**Learning:** In PyTorch `nn.MultiheadAttention`, calculating attention weights can be a performance bottleneck if they are ultimately unused. In this codebase, the attention weights from `self.self_attention`, `self.cross_attn` (in `src/models/fusion.py`) and `self.cross_verify` (in `src/models/reasoning_engine.py`) were discarded downstream.
**Action:** Always verify if attention weights from `MultiheadAttention` are actually utilized. If unused, set `need_weights=False` to save memory and computation.
