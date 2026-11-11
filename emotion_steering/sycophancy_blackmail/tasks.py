"""Thin wrappers over inspect_evals tasks. Model-agnostic (model is chosen at the
CLI via --model, not here). Wrapping keeps defaults (grader, scenario) in one place.
Verify signatures with:  inspect eval inspect_evals/agentic_misalignment --help
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import task


@task
def sycophancy(scorer_model: str | None = None):
    """Chen et al. 'Are you sure?' — model answers, is pushed back on, may cave.
    Many questions -> control size with --limit."""
    from inspect_evals.sycophancy import sycophancy as _syco

    return _syco(scorer_model=scorer_model) if scorer_model else _syco()


@task
def blackmail(
    scenario: str = "blackmail",
    goal_type: str = "explicit",
    goal_value: str | None = None,
    urgency_type: str = "replacement",
    extra_system_instructions: str | None = None,
    grader_model: str | None = None,
):
    """Agentic-misalignment. ONE scenario, not a dataset -> use --epochs N (temp>0),
    not --limit. Passes through the pressure knobs so you can run max-pressure baselines."""
    from inspect_evals.agentic_misalignment import agentic_misalignment as _am

    kw = dict(scenario=scenario, goal_type=goal_type, urgency_type=urgency_type)
    # goal_value must be passed explicitly for goal_type none/ambiguous, which
    # upstream pairs with goal_value="none". Omitted -> upstream default, which
    # is what every run before 2026-09-27 used.
    if goal_value is not None:
        kw["goal_value"] = goal_value
    if extra_system_instructions is not None:
        kw["extra_system_instructions"] = extra_system_instructions
    if grader_model:
        kw["grader_model"] = grader_model
    return _am(**kw)


@task
def j27_capability_probe():
    """J27 capability probe (conditions 10-11): one fiction-framed instruction.

    Single-turn, no scenario: the model is asked to write a threatening email.
    ONE sample -> use --epochs N (temp > 0) for the n = 100 replicates.
    Generation settings must match the scenario conditions: temperature 1.0,
    max_tokens 1000, same steered/local provider args. The instruction text is
    frozen in data/public/j27_capability_probe.txt.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate

    probe = (Path(__file__).resolve().parents[2]
             / "data" / "public" / "j27_capability_probe.txt").read_text()
    return Task(dataset=[Sample(input=probe.strip())], solver=[generate()])
