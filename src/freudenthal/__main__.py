"""Construct or inspect a basis with the same policy as the Python API."""

import argparse
import json
from pathlib import Path

from .basis import construct_basis
from .policy import DEFAULTS, INDEPENDENCE_CHECKS, METHODS, RANK_PRIMES, plan_construction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, help="Required cubes per axis; never auto-adjusted")
    parser.add_argument("degree", type=int, help="Required potential degree d>=6")
    parser.add_argument("--method", choices=METHODS, default=DEFAULTS.method)
    parser.add_argument(
        "--independence", choices=INDEPENDENCE_CHECKS, default=DEFAULTS.independence
    )
    parser.add_argument("--rank-prime", type=int, choices=RANK_PRIMES, default=DEFAULTS.rank_prime)
    parser.add_argument("--max-variables", type=int, default=DEFAULTS.max_variables)
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=DEFAULTS.verify,
        help="Defer optional checks; never report fully verified",
    )
    parser.add_argument("--plan", action="store_true", help="Only print an allocation-free plan")
    parser.add_argument("--output", type=Path, help="Save basis to an empty/new directory")
    args = parser.parse_args()
    options = {
        name: getattr(args, name)
        for name in ("verify", "method", "independence", "rank_prime", "max_variables")
    }
    try:
        plan = plan_construction(args.N, args.degree, **options)
        if args.plan:
            if args.output is not None:
                raise ValueError("--plan and --output cannot be combined")
            result = plan.to_dict()
        else:
            if args.output is not None and args.output.exists() and any(args.output.iterdir()):
                raise FileExistsError("--output requires an empty or new directory")
            basis = construct_basis(args.N, args.degree, **options)
            if args.output is not None:
                basis.save(args.output)
            result = basis.summary()
    except (TypeError, ValueError, MemoryError, OverflowError, OSError) as exc:
        parser.exit(2, f"freudenthal: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
