"""Return this PC's state and process liveness; never write state.json."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_generate.state_store import mutex, process_token, read_state


def snapshot(pc_id):
    with mutex():
        state = read_state()
        if state["pc_id"] != pc_id:
            raise ValueError("state.json belongs to another PC")
        for row in state["workers"]:
            if row["state"] == "running":
                token = process_token(row.get("pid"))
                row["process_alive"] = token is not None and token == row.get("process_token")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-id", required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.pc_id), ensure_ascii=True))


if __name__ == "__main__":
    main()
