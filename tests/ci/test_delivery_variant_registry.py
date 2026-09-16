# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python and C API delivery reports on one variant registry."""

from pathlib import Path

from tools.delivery_variants import load_delivery_variants


ROOT = Path(__file__).resolve().parents[2]


def test_delivery_registry_contains_the_current_40_unique_variant_ids():
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    assert len(variants) == 40
    assert len({variant["id"] for variant in variants}) == 40


def test_both_summary_writers_consume_the_shared_delivery_registry():
    python_runner = (ROOT / "run_flagsparse_pytest.py").read_text(encoding="utf-8")
    capi_writer = (ROOT / "capi" / "tools" / "write_summary.py").read_text(
        encoding="utf-8"
    )
    assert "load_delivery_variants" in python_runner
    assert "_delivery_results" in python_runner
    assert "load_delivery_variants" in capi_writer
    assert "expected_variants" in capi_writer
