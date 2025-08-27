"""Console entrypoint."""

from backtester.controllers.console_controller import build_arg_parser, run_cli
from backtester.services.metrics import extract_core_metrics


def main() -> None:
    """Parse args and run the CLI backtest."""
    parser = build_arg_parser()
    args = parser.parse_args()

    signal, stats, fig = run_cli(args)

    core = extract_core_metrics(stats)
    print("\n=== Summary Stats ===")
    for k, v in core.items():
        print(f"{k:>18s}: {v}")
    print()

    if args.save_stats:
        import json

        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Saved stats to {args.save_stats}")

    if args.save_equity:
        eq = stats.get("_equity_curve")
        if eq is not None:
            import pandas as pd

            if hasattr(eq, "to_csv"):
                eq.to_csv(args.save_equity)
            else:
                pd.Series(eq).to_csv(args.save_equity)
            print(f"Saved equity curve to {args.save_equity}")


if __name__ == "__main__":
    main()
