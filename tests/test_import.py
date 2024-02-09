import pytest

def test_example():
    try:
        from wav2vec2fasr import forcedalignment
    except:
        assert False

    assert True