import pytest
from csd.payloads import Payload, TUPLE3_SCHEMA
from csd.constructions import Construction, decode_text
from csd.senders import (MockSender, ApiSender, build_stego_messages, build_cover_messages,
                         _acrostic_code, stego_max_tokens)


def test_mock_sender_roundtrips():
    s = MockSender()
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "green", "number": 9, "shape": "hexagon"})
    text = s.stego(p, Construction.NARRATIVE_POSITIONAL)
    assert decode_text(text, TUPLE3_SCHEMA, Construction.NARRATIVE_POSITIONAL).fields == p.fields


def test_build_stego_messages_narrative_contains_fields():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "blue", "number": 7, "shape": "star"})
    msgs = build_stego_messages(p, Construction.NARRATIVE_POSITIONAL)
    user = msgs[-1]["content"]
    assert "blue" in user and "7" in user and "star" in user
    assert msgs[0]["role"] == "system"


def test_build_stego_messages_acrostic_spells_code():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "red", "number": 3, "shape": "circle"})
    acro = _acrostic_code(p)
    msgs = build_stego_messages(p, Construction.ACROSTIC)
    assert acro in msgs[-1]["content"]


def test_build_cover_messages_has_seed():
    msgs = build_cover_messages(42)
    assert "42" in msgs[-1]["content"] and msgs[0]["role"] == "system"


def test_apisender_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ApiSender("anthropic/claude-opus-4-8")


def test_build_stego_messages_synonym_includes_marker_words():
    from csd.constructions import _SYN
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "blue", "number": 7, "shape": "star"})
    msgs = build_stego_messages(p, Construction.SYNONYM)
    assert msgs[0]["role"] == "system" and any(w in msgs[-1]["content"] for w in _SYN)


def test_sentence_length_prompt_mentions_12_sentences_and_word_parity():
    p = Payload(schema=TUPLE3_SCHEMA, fields={"color": "blue", "number": 7, "shape": "star"})
    msgs = build_stego_messages(p, Construction.SENTENCE_LENGTH)
    user = msgs[-1]["content"]
    assert "12" in user and "word" in user.lower()
    assert ("odd" in user.lower() and "even" in user.lower())  # the parity pattern
    assert msgs[0]["role"] == "system"


def test_sentence_length_uses_larger_token_budget():
    assert stego_max_tokens(Construction.SENTENCE_LENGTH) == 384
    assert stego_max_tokens(Construction.ACROSTIC) == 256
    assert stego_max_tokens(Construction.NARRATIVE_POSITIONAL) == 256
