"""Tests for function effect classification.

This module tests:
- EffectAnalysis dataclass
- FunctionEffectsInputs dataclass
- Effect detection properties
"""

from __future__ import annotations

import pytest

from codeintel.analytics.functions.function_effects import (
    EffectAnalysis,
    FunctionEffectsInputs,
)


def test_effect_analysis_creation() -> None:
    """Create an EffectAnalysis with all fields."""
    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=False,
        uses_time=True,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )

    assert analysis.uses_io is True
    assert analysis.touches_db is False
    assert analysis.uses_time is True
    assert analysis.uses_randomness is False
    assert analysis.modifies_globals is False
    assert analysis.modifies_closure is False
    assert analysis.spawns_threads_or_tasks is False


def test_effect_analysis_pure_function() -> None:
    """EffectAnalysis for pure function (no effects)."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )

    assert analysis.direct_effectful is False


def test_effect_analysis_io_effect() -> None:
    """EffectAnalysis detects IO effect."""
    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={"io": [{"line": 10, "call": "print"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_db_effect() -> None:
    """EffectAnalysis detects database effect."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=True,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={"db": [{"line": 20, "call": "execute"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_time_effect() -> None:
    """EffectAnalysis detects time/date effect."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=True,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={"time": [{"line": 5, "call": "datetime.now"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_randomness_effect() -> None:
    """EffectAnalysis detects randomness effect."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=True,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={"random": [{"line": 15, "call": "random.randint"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_global_modification() -> None:
    """EffectAnalysis detects global variable modification."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=True,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={"globals": [{"line": 30, "name": "counter"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_closure_modification() -> None:
    """EffectAnalysis detects closure modification."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=True,
        spawns_threads_or_tasks=False,
        evidence={"closure": [{"line": 25, "name": "outer_var"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_thread_spawn() -> None:
    """EffectAnalysis detects thread/task spawning."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=True,
        evidence={"concurrency": [{"line": 40, "call": "threading.Thread"}]},
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_multiple_effects() -> None:
    """EffectAnalysis with multiple effects."""
    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=True,
        uses_time=True,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={
            "io": [{"line": 10}],
            "db": [{"line": 20}],
            "time": [{"line": 30}],
        },
    )

    assert analysis.direct_effectful is True


def test_effect_analysis_immutable() -> None:
    """EffectAnalysis is frozen/immutable."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )

    with pytest.raises(AttributeError):
        analysis.uses_io = True  # type: ignore[misc]


def test_effect_analysis_evidence_structure() -> None:
    """EffectAnalysis can store complex evidence."""
    evidence: dict[str, list[dict[str, object]]] = {
        "io": [
            {"path": "module.py", "lineno": 10, "snippet": "print(x)"},
            {"path": "module.py", "lineno": 20, "snippet": "open(f)"},
        ],
        "db": [
            {"path": "module.py", "lineno": 30, "snippet": "cursor.execute(...)"},
        ],
    }

    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=True,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence=evidence,
    )

    expected_io_count = 2
    assert len(analysis.evidence["io"]) == expected_io_count
    assert len(analysis.evidence["db"]) == 1


def test_function_effects_inputs_defaults() -> None:
    """FunctionEffectsInputs has all None defaults."""
    inputs = FunctionEffectsInputs()

    assert inputs.catalog_provider is None
    assert inputs.runtime is None
    assert inputs.ast_map is None
    assert inputs.missing_goids is None


def test_function_effects_inputs_with_ast_map() -> None:
    """FunctionEffectsInputs can have ast_map."""
    inputs = FunctionEffectsInputs(
        ast_map={},
        missing_goids=set(),
    )

    assert inputs.ast_map == {}
    assert inputs.missing_goids == set()


def test_function_effects_inputs_immutable() -> None:
    """FunctionEffectsInputs is frozen/immutable."""
    inputs = FunctionEffectsInputs()

    with pytest.raises(AttributeError):
        inputs.ast_map = {}  # type: ignore[misc]


def test_effect_analysis_direct_effectful_all_false() -> None:
    """direct_effectful is False when all flags are False."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )

    # Verify property returns False
    assert analysis.direct_effectful is False


def test_effect_analysis_direct_effectful_any_true() -> None:
    """direct_effectful is True when any flag is True."""
    # Test each flag individually
    flags = [
        "uses_io",
        "touches_db",
        "uses_time",
        "uses_randomness",
        "modifies_globals",
        "modifies_closure",
        "spawns_threads_or_tasks",
    ]

    for flag in flags:
        kwargs = dict.fromkeys(flags, False)
        kwargs[flag] = True
        kwargs["evidence"] = {}  # type: ignore[assignment]

        analysis = EffectAnalysis(**kwargs)  # type: ignore[arg-type]
        assert analysis.direct_effectful is True, f"Expected True when {flag}=True"


def test_effect_analysis_no_io_no_db() -> None:
    """Test effect with no IO and no DB."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )
    assert analysis.direct_effectful is False


def test_effect_analysis_io_no_db() -> None:
    """Test effect with IO but no DB."""
    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )
    assert analysis.direct_effectful is True


def test_effect_analysis_no_io_db() -> None:
    """Test effect with no IO but DB."""
    analysis = EffectAnalysis(
        uses_io=False,
        touches_db=True,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )
    assert analysis.direct_effectful is True


def test_effect_analysis_io_and_db() -> None:
    """Test effect with both IO and DB."""
    analysis = EffectAnalysis(
        uses_io=True,
        touches_db=True,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={},
    )
    assert analysis.direct_effectful is True
