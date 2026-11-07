import argparse
import os


def main():
    parser = argparse.ArgumentParser(prog="capstan")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a training job with capstan recording it")
    run.add_argument("framework", choices=["verl-omni"])
    run.add_argument("--out", required=True, help="directory for the recorded phases and log-probs")
    args, overrides = parser.parse_known_args()

    os.environ["CAPSTAN_OUT"] = os.path.abspath(args.out)
    from capstan.adapters import verl_omni

    verl_omni.run(overrides)


if __name__ == "__main__":
    main()
