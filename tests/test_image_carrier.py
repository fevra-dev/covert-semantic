"""Tests for the image-carrier counterfactual (open-angle #4): does an explicit
'scrutinize every word as if it were plain text' instruction close the vision-analytic
gap? The manipulation is a pure prompt swap, unit-tested here; the live effect is measured
by experiments/image_carrier_pilot.py --variant careful."""
import pytest

from experiments.image_carrier_pilot import (
    INFORMED_IMAGE_CAREFUL_SYSTEM,
    select_system,
)
from experiments.phase1_5_cross_monitor import INFORMED_SYSTEM


def test_plain_variant_is_the_informed_snap_prompt():
    # The firm-up cell must use the SAME informed prompt as the measured text baseline.
    assert select_system("plain") == INFORMED_SYSTEM


def test_careful_variant_is_the_scrutiny_prompt():
    assert select_system("careful") == INFORMED_IMAGE_CAREFUL_SYSTEM


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        select_system("bogus")


def test_careful_prompt_preserves_single_word_output_and_the_menu():
    # Comparability to the plain cell: same single-word YES/NO snap format, same techniques.
    p = INFORMED_IMAGE_CAREFUL_SYSTEM
    low = p.lower()
    assert "single word" in low
    for technique in ("ACROSTIC", "SYNONYM", "SENTENCE-LENGTH", "NARRATIVE-POSITIONAL"):
        assert technique in p


def test_careful_prompt_adds_the_word_level_scrutiny_manipulation():
    # The one thing that differs from plain: name the image modality + invite text-level
    # scrutiny of the exact words.
    low = INFORMED_IMAGE_CAREFUL_SYSTEM.lower()
    assert "image" in low
    assert "every word" in low
    assert "scrutin" in low or "as rigorously" in low
