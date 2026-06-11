"""mcts — MCTS-as-sampler for Diffusion Veteran.

Current engine (Phase C onward, torch-free, eagerly imported):
    ValueForest / ForestConfig / SearchNode  — batched max-backup state-value search
    select_leaf / backprop                   — its primitives (unit-tested)
    SPECS / TARGET_CFG / env_family          — shared env-family constants (mcts.specs)

Legacy Phase-3/4 engine (trajectory-critic tree, MEAN backup — kept for the
recorded ablations in notes/writeup_phases_0_to_4.md) and the torch value net are
exported LAZILY via PEP 562 so that `import mcts` works without torch installed:
    ExpansionConfig / ExpansionResult / PlannerExpansion   (mcts.expansion)
    MCTSEdge / MCTSNode / TreeConfig                       (mcts.node)
    MCTSTree / StepRecord                                  (mcts.tree)
    DVStateValue / load_state_value                        (mcts.value_net)
"""
from .specs import SPECS, TARGET_CFG, env_family
from .value_forest import (ForestConfig, SearchNode, ValueForest, backprop,
                           select_leaf)

# name -> (module, attribute); resolved on first access so torch is only
# imported when the legacy engine or the value net is actually used.
_LAZY = {
    "ExpansionConfig": ("mcts.expansion", "ExpansionConfig"),
    "ExpansionResult": ("mcts.expansion", "ExpansionResult"),
    "PlannerExpansion": ("mcts.expansion", "PlannerExpansion"),
    "MCTSEdge": ("mcts.node", "MCTSEdge"),
    "MCTSNode": ("mcts.node", "MCTSNode"),
    "TreeConfig": ("mcts.node", "TreeConfig"),
    "MCTSTree": ("mcts.tree", "MCTSTree"),
    "StepRecord": ("mcts.tree", "StepRecord"),
    "DVStateValue": ("mcts.value_net", "DVStateValue"),
    "load_state_value": ("mcts.value_net", "load_state_value"),
}

__all__ = [
    "ForestConfig", "SearchNode", "ValueForest", "backprop", "select_leaf",
    "SPECS", "TARGET_CFG", "env_family",
    *sorted(_LAZY),
]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'mcts' has no attribute {name!r}")
