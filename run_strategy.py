import argparse

from functions.pipeline_orchestrator import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Stock Strategy")
    parser.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--tickers", type=str, default=None, nargs="+", help="List of tickers")
    parser.add_argument("--s3-bucket", type=str, help="S3 Bucket Name")
    parser.add_argument("--input-prefix", type=str, help="Input folder path in S3")

    args = parser.parse_args()

    run_full_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
        s3_bucket=args.s3_bucket,
        input_prefix=args.input_prefix,
    )


if __name__ == "__main__":
    main()
