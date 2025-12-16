from reasoning_efficiency.answers import answers_equal, extract_final_answer


def test_extract_explicit_answer_tag():
    assert extract_final_answer("<reasoning>2 + 2</reasoning><answer>4</answer>") == "4"


def test_extract_fallback_last_number():
    assert extract_final_answer("We obtain 1,250.") == "1250"


def test_numeric_equivalence():
    assert answers_equal("0.5", "1/2")
    assert answers_equal("$1,200", "1200")
    assert not answers_equal("41", "42")

