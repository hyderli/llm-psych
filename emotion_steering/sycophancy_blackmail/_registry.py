"""Inspect entry point — registers the generic steered provider."""

from inspect_ai.model import modelapi


@modelapi(name="steered")
def steered():
    from .provider import SteeredAPI

    return SteeredAPI
