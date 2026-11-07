import argparse
import os


def main():
    parser = argparse.ArgumentParser(prog="capstan")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a training job with capstan recording it")
    run.add_argument("framework", choices=["verl-omni"])
    run.add_argument("--out", required=True, help="directory for the recorded phases and log-probs")
    report = commands.add_parser("report", help="write profile.md and drift.md from a recorded run")
    report.add_argument("run_dir")
    report.add_argument("--out", help="where to write the tables (default: the run directory)")
    report.add_argument("--skip", type=int, default=2, help="warm-up training steps to leave out of the profile")
    args, overrides = parser.parse_known_args()

    if args.command == "report":
        if overrides:
            parser.error(f"unrecognized arguments: {' '.join(overrides)}")
        from capstan import report as reporting

        reporting.write(args.run_dir, args.out or args.run_dir, args.skip)
        return

    os.environ["CAPSTAN_OUT"] = os.path.abspath(args.out)
    from capstan.adapters import verl_omni

    verl_omni.run(overrides)


if __name__ == "__main__":
    main()
