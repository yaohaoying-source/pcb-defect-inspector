import argparse
import csv
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "pcb_defect_inspector"
LIMIT = 30

sys.path.insert(0, str(APP_DIR))
from upload.uploader import upload_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Upload a sample of a previous benchmark run")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output CSV, do not point to the backup")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    result_dir = args.result_dir
    rows = []
    with (result_dir / "batch_detection_result.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("result_image"):
                rows.append(row)
            if len(rows) >= LIMIT:
                break

    out_rows = []
    for row in rows:
        path = result_dir / "detect_images" / row["result_image"]
        t0 = time.perf_counter()
        try:
            upload_file(str(path), args.host, args.port)
            success = True
            status = "ok"
        except Exception as exc:
            success = False
            status = str(exc)
        out_rows.append(
            {
                "image": row["image"],
                "result_image": row["result_image"],
                "bytes": path.stat().st_size if path.exists() else "",
                "upload_success": success,
                "upload_ms": round((time.perf_counter() - t0) * 1000, 3),
                "upload_status": status,
            }
        )
        print(f"{len(out_rows)}/{len(rows)} {path.name} {success} {status}")

    with args.output.open("x", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["image", "result_image", "bytes", "upload_success", "upload_ms", "upload_status"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"success={sum(1 for row in out_rows if row['upload_success'])}/{len(out_rows)}")


if __name__ == "__main__":
    main()
