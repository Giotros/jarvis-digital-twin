"""Tests for the Greek PII sanitizer — including the v2 final-sigma (ς) bug."""

import pytest

from jarvis.sanitization import PIISanitizer
from jarvis.sanitization.patterns import is_valid_afm, is_valid_amka


def make_valid_afm() -> str:
    """Construct a 9-digit ΑΦΜ that satisfies the mod-11 checksum."""
    base = "12345678"
    check = (sum(int(d) * 2 ** (8 - i) for i, d in enumerate(base)) % 11) % 10
    return base + str(check)


@pytest.fixture()
def sanitizer() -> PIISanitizer:
    return PIISanitizer(contacts=["Γιώργος Παπαδόπουλος", "Νίκος", "Μαρία", "Κώστας"])


# ---------------------------------------------------------------- names / ς

def test_final_sigma_names_are_masked(sanitizer):
    """THE v2 BUG: names ending in final sigma escaped detection."""
    out = sanitizer.sanitize("Ο Γιώργος και ο Νίκος ήρθαν χθες.")
    assert "Γιώργος" not in out and "Νίκος" not in out
    assert out.count("[ΟΝΟΜΑ]") == 2


def test_declension_variants_masked(sanitizer):
    out = sanitizer.sanitize("Πες του Γιώργου να πάρει τον Κώστα τηλέφωνο.")
    assert "Γιώργου" not in out and "Κώστα" not in out
    assert out.count("[ΟΝΟΜΑ]") == 2


def test_lowercase_and_unaccented_matched(sanitizer):
    out = sanitizer.sanitize("ο γιωργος είπε οκ")
    assert "γιωργος" not in out


def test_full_name_pattern_masks_unknown_names(sanitizer):
    out = sanitizer.sanitize("Μίλησα με τον Πέτρο Αλεξίου το πρωί.")
    assert "Αλεξίου" not in out


def test_non_names_not_masked(sanitizer):
    text = "Καλημέρα! Ευχαριστώ πολύ. Καλό Πάσχα!"
    assert sanitizer.sanitize(text) == text


# ---------------------------------------------------------------- financial

def test_valid_afm_masked_invalid_kept(sanitizer):
    valid = make_valid_afm()
    assert is_valid_afm(valid)
    out = sanitizer.sanitize(f"Το ΑΦΜ μου είναι {valid}")
    assert valid not in out and "[ΑΦΜ]" in out
    # 9 digits failing the checksum are NOT PII (e.g. an order number)
    invalid = valid[:-1] + str((int(valid[-1]) + 1) % 10)
    assert invalid in sanitizer.sanitize(f"Κωδικός παραγγελίας {invalid}")


def test_amka_date_validation(sanitizer):
    assert is_valid_amka("01018012345")
    out = sanitizer.sanitize("ΑΜΚΑ: 01018012345")
    assert "[ΑΜΚΑ]" in out
    assert "99999912345" in sanitizer.sanitize("σειριακός 99999912345")


def test_greek_iban_masked(sanitizer):
    out = sanitizer.sanitize("Στείλε στο GR1601101250000000012300695 ευρώ")
    assert "[IBAN]" in out and "GR16" not in out


# ---------------------------------------------------------------- phones / email

@pytest.mark.parametrize("phone", ["6912345678", "+30 6912345678", "2101234567"])
def test_phones_masked(sanitizer, phone):
    out = sanitizer.sanitize(f"πάρε με στο {phone} το βράδυ")
    assert phone not in out and "[ΤΗΛΕΦΩΝΟ]" in out


def test_email_masked(sanitizer):
    out = sanitizer.sanitize("στείλε στο test.user@example.gr")
    assert "[EMAIL]" in out and "example.gr" not in out


# ---------------------------------------------------------------- records API

def test_sanitize_records_report(sanitizer):
    records = [
        {"instruction": "Πάρε τον Νίκο στο 6912345678", "response": "οκ θα τον πάρω"},
        {"instruction": "τι ώρα;", "response": "στις 9"},
    ]
    clean, report = sanitizer.sanitize_records(records)
    assert report.records_processed == 2
    assert report.records_changed == 1
    assert report.total_replacements == 2          # name + phone
    assert "6912345678" not in clean[0]["instruction"]
    assert records[0]["instruction"] != clean[0]["instruction"]   # no mutation


def test_refuses_overwriting_raw_file(tmp_path, sanitizer):
    src = tmp_path / "raw.json"
    src.write_text('[{"instruction": "a", "response": "b"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        sanitizer.sanitize_json_file(src, src)
