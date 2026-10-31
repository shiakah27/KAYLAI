"""Options and safety checks shared by the run scripts.

--from-cache  replays recorded responses only and stops at the first cache miss.
--route       sends gpt-oss calls to Groq directly or through OpenRouter pinned to Groq.
              The route is part of the cache key, so replaying a run needs the route it used.

A run whose calls failed must not overwrite the committed results, so scripts call
`check_no_failures` before writing anything to results/."""

import argparse
import copy
from typing import Callable, Iterable

from kayl.client import CacheMissError, CallResult


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-cache", action="store_true",
                        help="use recorded responses only; stop on any cache miss (no API key needed)")
    parser.add_argument("--route", choices=["groq", "openrouter"],
                        help="route for gpt-oss calls (default: routing.gpt_oss in config.yaml)")


def apply_run_args(cfg: dict, args: argparse.Namespace) -> dict:
    """A copy of the config with the command-line options applied. config.yaml is not changed."""
    cfg = copy.deepcopy(cfg)
    cfg["cache_only"] = args.from_cache
    if args.route:
        cfg["routing"]["gpt_oss"] = args.route
    return cfg


def failed(results: Iterable[CallResult]) -> list[str]:
    """Errors of calls that failed. A task skipped because it does not fit a context window
    is an expected outcome, not a failure."""
    return [f"{r.task_id} {r.model}: {r.error}" for r in results if r.error and not r.skipped]


def check_no_failures(errors: list[str], stopped: list[str] = ()) -> None:
    """Exit before writing results if any call failed or the run stopped early."""
    if not errors and not stopped:
        return
    lines = [f"  stopped: {s}" for s in stopped] + [f"  failed: {e}" for e in errors[:10]]
    if len(errors) > 10:
        lines.append(f"  ... and {len(errors) - 10} more")
    raise SystemExit(f"{len(errors)} failed call(s), {len(stopped)} early stop(s); "
                     "results/ was NOT written. Fix the cause and rerun (finished calls are cached).\n"
                     + "\n".join(lines))


def run(main: Callable[[], None]) -> None:
    """Run a script's main and turn a cache miss into a clear exit message."""
    try:
        main()
    except CacheMissError as e:
        raise SystemExit(f"{e}\nresults/ was NOT written.") from None
