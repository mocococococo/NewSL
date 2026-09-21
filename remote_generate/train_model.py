import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from learning_param import BATCH_SIZE, EPOCHS
from transformer.learn import train
from remote_config import (
    get_data_root,
    get_model_root,
    get_record_root,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--shot",
        type=int,
        required=True,
    )
    
    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--model-name",
        default=None,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    data_dir = (
        get_data_root(
            ROOT,
            args.run_name,
        )
        / f"end{args.end}"
        / f"shot{args.shot}"
    )

    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory does not exist: {data_dir}"
        )

    data_files = sorted(
        data_dir.glob("sl_data_*.npz")
    )

    if not data_files:
        raise SystemExit(
            f"No training data found: {data_dir}"
        )

    model_name = (
        args.model_name
        if args.model_name is not None
        else f"shot-end{args.end}-shot{args.shot}"
    )

    temporary_model_path = (
        ROOT
        / "model"
        / f"{model_name}.bin"
    )

    output_model_path = (
        get_model_root(
            ROOT,
            args.run_name,
        )
        / f"end_{args.end}"
        / f"{model_name}.bin"
    )

    print("TRAINING")
    print("RUN:", args.run_name)
    print("END:", args.end)
    print("SHOT:", args.shot)
    print("DATA:", data_dir)
    print("FILES:", len(data_files))
    print("BATCH SIZE:", args.batch_size)
    print("EPOCHS:", args.epochs)
    print("MODEL:", model_name)

    train(
        program_dir=ROOT,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_name=model_name,
        use_gpu=not args.cpu,
        data_dir=data_dir,
        loss_history_dir=(
            get_record_root(
                ROOT,
                args.run_name,
            )
            / f"end_{args.end}"
        ),
    )


    if not temporary_model_path.is_file():
        raise FileNotFoundError(
            f"Training finished but model was not created: "
            f"{temporary_model_path}"
        )

    output_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_model_path.replace(
        output_model_path
    )

    print(
        f"Moved model to: "
        f"{output_model_path}"
    )

if __name__ == "__main__":
    main()