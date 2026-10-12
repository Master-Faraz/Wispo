from wispo.asr.accumulator import TranscriptAccumulator, merge_transcript


def test_growing_hypothesis_replaces_the_shorter_one():
    assert merge_transcript("hello", "hello this is") == "hello this is"
    assert merge_transcript("hello this is", "hello this is a test") == "hello this is a test"


def test_overlap_is_not_repeated():
    merged = merge_transcript("Create a new function", "function called calculate")
    assert merged == "Create a new function called calculate"


def test_committed_and_partial_follow_the_prd_split():
    acc = TranscriptAccumulator()
    acc.merge("Create a new function")
    assert acc.committed == ""
    assert acc.partial == "Create a new function"

    acc.merge("function called calculate")
    assert acc.committed == "Create a new function"
    assert acc.partial == "called calculate"
    assert acc.text == "Create a new function called calculate"

    assert acc.finish() == "Create a new function called calculate"
    assert acc.partial == ""
