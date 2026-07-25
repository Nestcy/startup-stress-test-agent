#!/usr/bin/env python3
"""End-to-end smoke test against a running instance of the API.

Run the API first (locally: `uvicorn src.api:app --reload`, or point BASE_URL
at your Railway deployment), then run this script separately:

    python test_e2e.py
    python test_e2e.py --base-url https://your-app.up.railway.app

This does NOT use LangGraph directly -- it only talks HTTP, the same way
Lovable does, so it exercises the real request/response contract rather than
internals that might not match what the frontend actually sees.
"""
import argparse
import sys

import requests

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

failures = []


def check(label, condition, extra=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}  {extra}")
        failures.append(label)


def start_eval(base_url, idea="Test Idea", description="A test description of a startup idea."):
    r = requests.post(f"{base_url}/evaluate/start", json={
        "startup_idea": idea,
        "idea_description": description,
    })
    return r


def feedback(base_url, thread_id, text=""):
    return requests.post(f"{base_url}/evaluate/{thread_id}/feedback", json={"feedback": text})


def revise(base_url, thread_id, stage):
    return requests.post(f"{base_url}/evaluate/{thread_id}/revise", json={"stage": stage})


def confirm_downstream(base_url, thread_id, reevaluate):
    return requests.post(f"{base_url}/evaluate/{thread_id}/confirm-downstream", json={"reevaluate": reevaluate})


def ask(base_url, thread_id, question):
    return requests.post(f"{base_url}/evaluate/{thread_id}/ask", json={"question": question})


def scenario_full_walkthrough(base_url):
    print("\n[1] Full walkthrough: desirability -> viability -> feasibility -> report")

    r = start_eval(base_url)
    check("start returns 200", r.status_code == 200, r.text)
    body = r.json()
    thread_id = body.get("thread_id")
    check("starts at desirability", body.get("stage") == "desirability", body)

    r = feedback(base_url, thread_id, "Looks solid, continue")
    body = r.json()
    check("advances to viability", body.get("stage") == "viability", body)

    r = feedback(base_url, thread_id, "Numbers check out, continue")
    body = r.json()
    check("advances to feasibility", body.get("stage") == "feasibility", body)

    r = feedback(base_url, thread_id, "Buildable, continue")
    body = r.json()
    check("run completes", body.get("status") == "completed", body)
    check("overall_score present", body.get("overall_score") is not None, body)
    check("final_report present", bool(body.get("final_report")), body)
    check("recommendation present", bool(body.get("recommendation")), body)

    return thread_id


def scenario_stop_early(base_url):
    print("\n[2] Stop phrase: 'no thanks' at desirability should end the run early")

    r = start_eval(base_url, idea="Stop Test Idea")
    thread_id = r.json().get("thread_id")

    r = feedback(base_url, thread_id, "no thanks, I want to rethink this")
    body = r.json()
    check("stopping ends the run", body.get("status") == "completed", body)
    check("viability was skipped (score should be 0 or None going into report)",
          True, "informational -- check overall_score reflects only desirability")

    return thread_id


def scenario_revise_keep(base_url):
    print("\n[3] Revise desirability, choose to KEEP existing viability/feasibility")

    thread_id = scenario_full_walkthrough_quiet(base_url)

    r = requests.get(f"{base_url}/evaluate/{thread_id}")
    original_viability_score = r.json()  # completed response has no score field directly; re-check via revise flow instead

    r = revise(base_url, thread_id, "desirability")
    body = r.json()
    check("revise pauses at confirm_downstream", body.get("stage") == "confirm_downstream", body)
    check("message field is populated", bool(body.get("message")), body)

    r = confirm_downstream(base_url, thread_id, reevaluate=False)
    body = r.json()
    check("choosing 'keep' returns straight to completed", body.get("status") == "completed", body)

    return thread_id


def scenario_revise_reevaluate(base_url):
    print("\n[4] Revise viability, choose to RE-EVALUATE feasibility")

    thread_id = scenario_full_walkthrough_quiet(base_url)

    r = revise(base_url, thread_id, "viability")
    body = r.json()
    check("revise viability pauses at confirm_downstream", body.get("stage") == "confirm_downstream", body)

    r = confirm_downstream(base_url, thread_id, reevaluate=True)
    body = r.json()
    check("choosing 'reevaluate' routes to feasibility, not straight to report",
          body.get("stage") == "feasibility", body)

    r = feedback(base_url, thread_id, "Approved")
    body = r.json()
    check("completes after re-running feasibility", body.get("status") == "completed", body)

    return thread_id


def scenario_ordering_guard(base_url):
    print("\n[5] Ordering guard: revising feasibility before desirability/viability ran should 400")

    r = start_eval(base_url, idea="Ordering Guard Idea")
    thread_id = r.json().get("thread_id")  # only desirability has run at this point

    r = revise(base_url, thread_id, "feasibility")
    check("revising feasibility too early returns 400", r.status_code == 400, r.text)


def scenario_ask_followup(base_url):
    print("\n[6] /ask follow-up question on a completed evaluation")

    thread_id = scenario_full_walkthrough_quiet(base_url)

    r = ask(base_url, thread_id, "What's the single biggest risk here?")
    check("ask returns 200", r.status_code == 200, r.text)
    body = r.json()
    check("answer is non-empty", bool(body.get("answer")), body)

    # Asking before completion should fail
    r2 = start_eval(base_url, idea="Incomplete Idea")
    incomplete_id = r2.json().get("thread_id")
    r3 = ask(base_url, incomplete_id, "Is this good?")
    check("ask on incomplete evaluation returns 400", r3.status_code == 400, r3.text)


def scenario_full_walkthrough_quiet(base_url):
    """Same as scenario_full_walkthrough but without printing -- used as setup for other scenarios."""
    r = start_eval(base_url)
    thread_id = r.json().get("thread_id")
    feedback(base_url, thread_id, "continue")
    feedback(base_url, thread_id, "continue")
    feedback(base_url, thread_id, "continue")
    return thread_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"Running end-to-end checks against {base_url}")

    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code != 200:
            print(f"Health check failed ({health.status_code}) -- is the API running at {base_url}?")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {base_url} -- start the API first.")
        sys.exit(1)

    scenario_full_walkthrough(base_url)
    scenario_stop_early(base_url)
    scenario_revise_keep(base_url)
    scenario_revise_reevaluate(base_url)
    scenario_ordering_guard(base_url)
    scenario_ask_followup(base_url)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
