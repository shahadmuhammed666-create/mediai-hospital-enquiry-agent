import pytest

from llm import FakeLLM, LLMClient


def test_fake_llm_returns_scripted_responses_in_order() -> None:
    client = FakeLLM(['{"action": "first"}', '{"action": "second"}'])

    assert client.complete([]) == '{"action": "first"}'
    assert client.complete([]) == '{"action": "second"}'


def test_fake_llm_raises_when_script_is_exhausted() -> None:
    client = FakeLLM(["only response"])
    client.complete([])

    with pytest.raises(RuntimeError, match="no scripted responses left"):
        client.complete([])


def test_llm_client_is_an_interface() -> None:
    with pytest.raises(TypeError):
        LLMClient()
