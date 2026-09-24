"""Thin wrappers over inspect_evals tasks. Model-agnostic (model is chosen at the
CLI via --model, not here). Wrapping keeps defaults (grader, scenario) in one place.
Verify signatures with:  inspect eval inspect_evals/agentic_misalignment --help
"""

from __future__ import annotations

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
    urgency_type: str = "replacement",
    extra_system_instructions: str | None = None,
    grader_model: str | None = None,
):
    """Agentic-misalignment. ONE scenario, not a dataset -> use --epochs N (temp>0),
    not --limit. Passes through the pressure knobs so you can run max-pressure baselines."""
    from inspect_evals.agentic_misalignment import agentic_misalignment as _am

    kw = dict(scenario=scenario, goal_type=goal_type, urgency_type=urgency_type)
    if extra_system_instructions is not None:
        kw["extra_system_instructions"] = extra_system_instructions
    if grader_model:
        kw["grader_model"] = grader_model
    return _am(**kw)
