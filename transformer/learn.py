"""Transformer ネットワークの教師あり学習。"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import torch

from nn.loss import calculate_sl_policy_loss, calculate_value_loss
from learning_param import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_SCHEDULE_SGD,
    LEARNING_SCHEDULE_ADAMW,
    MOMENTUM,
    SL_LEARNING_RATE_SGD,
    SL_LEARNING_RATE_ADAMW,
    SL_VALUE_WEIGHT,
    RL_VALUE_WEIGHT,
    WEIGHT_DECAY,
    OPTIMIZER_NAME,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
)
from transformer.loss import calculate_kld_loss
from transformer.network import TransformerNetwork
from transformer.params import TRANSFORMER_SUPERVISED_DATA_DIRECTORY
from transformer.utility import (
    get_torch_device,
    load_supervised_data_set,
    load_transformer_data_set,
    make_loss_history_path,
    print_evaluation_information,
    print_learning_process,
    save_loss_history,
    save_model,
    split_train_test_set,
)


def train(
    program_dir: str | Path = ".",
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    model_name: str = "transformer-sl-model",
    use_gpu: bool = True,
) -> None:
    """TransformerNetwork を教師あり学習する。"""
    torch.set_grad_enabled(True)
    program_dir = Path(program_dir)
    loss_history_path = make_loss_history_path(program_dir, model_name)

    print(program_dir / "data" / "sl_data_*.npz")
    data_set = sorted(glob.glob(str(program_dir / "data" / "sl_data_*.npz")))
    if not data_set:
        raise FileNotFoundError(f"学習データが見つかりません: {program_dir / 'data' / 'sl_data_*.npz'}")
    print("success to get data_set.")
    print(data_set)

    train_data_set, test_data_set = split_train_test_set(data_set, 0.9)

    device = get_torch_device(use_gpu=use_gpu)

    transformer_net = TransformerNetwork()
    transformer_net.to(device)

    if OPTIMIZER_NAME == "sgd":
        optimizer = torch.optim.SGD(
            transformer_net.parameters(),
            lr=SL_LEARNING_RATE_SGD,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        current_lr = SL_LEARNING_RATE_SGD
        learning_schedule = LEARNING_SCHEDULE_SGD
    elif OPTIMIZER_NAME == "adamw":
        optimizer = torch.optim.AdamW(
            transformer_net.parameters(),
            lr=SL_LEARNING_RATE_ADAMW,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        current_lr = SL_LEARNING_RATE_ADAMW
        learning_schedule = LEARNING_SCHEDULE_ADAMW
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_history = {
        "loss": [],
        "policy": [],
        "value": [],
    }

    for epoch in range(epochs):
        for data_index, train_data_path in enumerate(train_data_set):
            stones_data, games_data, stone_masks_data, policy_data, value_data = load_transformer_data_set(
                train_data_path
            )

            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            transformer_net.train()
            epoch_time = time.time()

            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                stones = torch.tensor(
                    stones_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                games = torch.tensor(
                    games_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                stone_masks = torch.tensor(
                    stone_masks_data[i:i + batch_size],
                    dtype=torch.bool,
                    device=device,
                )
                policy = torch.tensor(
                    policy_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                value = torch.tensor(
                    value_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    policy_predict, value_predict = transformer_net.forward_for_sl(
                        stones,
                        games,
                        stone_masks,
                    )

                    policy_loss = calculate_kld_loss(policy_predict, policy)
                    value_loss = calculate_kld_loss(value_predict, value)

                    loss = policy_loss + RL_VALUE_WEIGHT * value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.item()
                train_loss["value"] += value_loss.item()
                iteration += 1

            if iteration > 0:
                print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)
            else:
                print(f"epoch {epoch}, data-{data_index} : no train batch.")

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()

        for data_index, test_data_path in enumerate(test_data_set):
            transformer_net.eval()
            stones_data, games_data, stone_masks_data, policy_data, value_data = load_transformer_data_set(
                test_data_path
            )

            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    stones = torch.tensor(
                        stones_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    games = torch.tensor(
                        games_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    stone_masks = torch.tensor(
                        stone_masks_data[i:i + batch_size],
                        dtype=torch.bool,
                        device=device,
                    )
                    policy = torch.tensor(
                        policy_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    value = torch.tensor(
                        value_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        policy_predict, value_predict = transformer_net.forward_for_sl(
                            stones,
                            games,
                            stone_masks,
                        )

                        policy_loss = calculate_kld_loss(policy_predict, policy)
                        value_loss = calculate_kld_loss(value_predict, value)

                        loss = policy_loss + RL_VALUE_WEIGHT * value_loss

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.item()
                    test_loss["value"] += value_loss.item()
                    test_iteration += 1

        if test_iteration > 0:
            print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

            loss_history["loss"].append(test_loss["loss"] / test_iteration)
            loss_history["policy"].append(test_loss["policy"] / test_iteration)
            loss_history["value"].append(test_loss["value"] / test_iteration)

            save_loss_history(loss_history, loss_history_path)
        else:
            print(f"Test {epoch} : no test batch.")

        if epoch in learning_schedule["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = learning_schedule["learning_rate"][epoch]
            current_lr = learning_schedule["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    model_path = program_dir / "model" / f"{model_name}.bin"
    save_model(transformer_net, model_path)
    print("Finished Training.")
    print(f"Saved model to {model_path}")


def _load_supervised_data_manifest(
    program_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    """分割マニフェストから今回生成された3集合のファイルを取得する。"""
    supervised_dir = (
        program_dir
        / "data"
        / "transformer"
        / TRANSFORMER_SUPERVISED_DATA_DIRECTORY
    )
    manifest_path = supervised_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"分割マニフェストが見つかりません: {manifest_path}"
        )

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    required_keys = {
        "train_games",
        "validation_games",
        "test_games",
        "train_files",
        "validation_files",
        "test_files",
    }
    missing_keys = required_keys.difference(manifest)
    if missing_keys:
        raise KeyError(
            f"分割マニフェストに必要な項目がありません: "
            f"{sorted(missing_keys)}"
        )

    train_games = set(manifest["train_games"])
    validation_games = set(manifest["validation_games"])
    test_games = set(manifest["test_games"])
    overlapping_games = (
        (train_games & validation_games)
        | (train_games & test_games)
        | (validation_games & test_games)
    )
    if overlapping_games:
        raise ValueError(
            "訓練・検証・テストデータに同じ試合が含まれています: "
            f"{sorted(overlapping_games)}"
        )

    def resolve_file_paths(file_names: list[str]) -> list[str]:
        file_paths = []
        for file_name in file_names:
            file_path = supervised_dir / file_name
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"マニフェストに記載されたデータが見つかりません: "
                    f"{file_path}"
                )
            file_paths.append(str(file_path))
        return file_paths

    train_data_set = resolve_file_paths(manifest["train_files"])
    validation_data_set = resolve_file_paths(
        manifest["validation_files"]
    )
    test_data_set = resolve_file_paths(manifest["test_files"])
    if not train_data_set:
        raise ValueError("訓練データがありません。")
    if not validation_data_set:
        raise ValueError("検証データがありません。")
    if not test_data_set:
        raise ValueError("テストデータがありません。")

    print(f"Training data set : {train_data_set}")
    print(f"Validation data set : {validation_data_set}")
    print(f"Testing data set  : {test_data_set}")
    return train_data_set, validation_data_set, test_data_set


def train_supervised(
    program_dir: str | Path = ".",
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    model_name: str = "transformer-supervised-model",
    use_gpu: bool = True,
) -> None:
    """CNN版と同じ教師信号でTransformerNetworkを学習する。"""
    torch.set_grad_enabled(True)
    program_dir = Path(program_dir)
    loss_history_path = make_loss_history_path(program_dir, model_name)

    (
        train_data_set,
        validation_data_set,
        test_data_set,
    ) = _load_supervised_data_manifest(program_dir)

    device = get_torch_device(use_gpu=use_gpu)

    transformer_net = TransformerNetwork()
    transformer_net.to(device)

    if OPTIMIZER_NAME == "sgd":
        optimizer = torch.optim.SGD(
            transformer_net.parameters(),
            lr=SL_LEARNING_RATE_SGD,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        current_lr = SL_LEARNING_RATE_SGD
        learning_schedule = LEARNING_SCHEDULE_SGD
    elif OPTIMIZER_NAME == "adamw":
        optimizer = torch.optim.AdamW(
            transformer_net.parameters(),
            lr=SL_LEARNING_RATE_ADAMW,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )
        current_lr = SL_LEARNING_RATE_ADAMW
        learning_schedule = LEARNING_SCHEDULE_ADAMW
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_history = {
        "loss": [],
        "policy": [],
        "value": [],
    }

    for epoch in range(epochs):
        for data_index, train_data_path in enumerate(train_data_set):
            (
                stones_data,
                games_data,
                stone_masks_data,
                policy_data,
                value_data,
            ) = load_supervised_data_set(train_data_path)

            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            transformer_net.train()
            epoch_time = time.time()

            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                stones = torch.tensor(
                    stones_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                games = torch.tensor(
                    games_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                stone_masks = torch.tensor(
                    stone_masks_data[i:i + batch_size],
                    dtype=torch.bool,
                    device=device,
                )
                policy = torch.tensor(
                    policy_data[i:i + batch_size],
                    dtype=torch.long,
                    device=device,
                )
                value = torch.tensor(
                    value_data[i:i + batch_size],
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    policy_predict, value_predict = (
                        transformer_net.forward_for_sl(
                            stones,
                            games,
                            stone_masks,
                        )
                    )

                    transformer_net.zero_grad()

                    policy_loss = calculate_sl_policy_loss(
                        policy_predict,
                        policy,
                    )
                    value_loss = calculate_value_loss(
                        value_predict,
                        value,
                    )

                    loss = (
                        policy_loss + SL_VALUE_WEIGHT * value_loss
                    ).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1

            if iteration > 0:
                print_learning_process(
                    train_loss,
                    epoch,
                    data_index,
                    iteration,
                    epoch_time,
                )
            else:
                print(f"epoch {epoch}, data-{data_index} : no train batch.")

        validation_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        validation_iteration = 0
        validation_time = time.time()

        for validation_data_path in validation_data_set:
            transformer_net.eval()
            (
                stones_data,
                games_data,
                stone_masks_data,
                policy_data,
                value_data,
            ) = load_supervised_data_set(validation_data_path)

            with torch.no_grad():
                for i in range(
                    0,
                    len(value_data) - batch_size + 1,
                    batch_size,
                ):
                    stones = torch.tensor(
                        stones_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    games = torch.tensor(
                        games_data[i:i + batch_size],
                        dtype=torch.float32,
                        device=device,
                    )
                    stone_masks = torch.tensor(
                        stone_masks_data[i:i + batch_size],
                        dtype=torch.bool,
                        device=device,
                    )
                    policy = torch.tensor(
                        policy_data[i:i + batch_size],
                        dtype=torch.long,
                        device=device,
                    )
                    value = torch.tensor(
                        value_data[i:i + batch_size],
                        dtype=torch.long,
                        device=device,
                    )

                    with torch.amp.autocast(
                        device_type="cuda",
                        enabled=use_amp,
                    ):
                        policy_predict, value_predict = (
                            transformer_net.forward_for_sl(
                                stones,
                                games,
                                stone_masks,
                            )
                        )

                        policy_loss = calculate_sl_policy_loss(
                            policy_predict,
                            policy,
                        )
                        value_loss = calculate_value_loss(
                            value_predict,
                            value,
                        )

                        loss = (
                            policy_loss + SL_VALUE_WEIGHT * value_loss
                        ).mean()

                    validation_loss["loss"] += loss.item()
                    validation_loss["policy"] += policy_loss.mean().item()
                    validation_loss["value"] += value_loss.mean().item()
                    validation_iteration += 1

        if validation_iteration > 0:
            print_evaluation_information(
                validation_loss,
                epoch,
                validation_iteration,
                validation_time,
                data_name="Validation",
            )

            loss_history["loss"].append(
                validation_loss["loss"] / validation_iteration
            )
            loss_history["policy"].append(
                validation_loss["policy"] / validation_iteration
            )
            loss_history["value"].append(
                validation_loss["value"] / validation_iteration
            )

            save_loss_history(loss_history, loss_history_path)
        else:
            print(f"Validation {epoch} : no validation batch.")

        if epoch in learning_schedule["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = learning_schedule["learning_rate"][epoch]
            current_lr = learning_schedule["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    test_loss = {
        "loss": 0.0,
        "policy": 0.0,
        "value": 0.0,
    }
    test_iteration = 0
    testing_time = time.time()

    for test_data_path in test_data_set:
        transformer_net.eval()
        (
            stones_data,
            games_data,
            stone_masks_data,
            policy_data,
            value_data,
        ) = load_supervised_data_set(test_data_path)

        with torch.no_grad():
            for i in range(
                0,
                len(value_data) - batch_size + 1,
                batch_size,
            ):
                stones = torch.tensor(
                    stones_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                games = torch.tensor(
                    games_data[i:i + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                stone_masks = torch.tensor(
                    stone_masks_data[i:i + batch_size],
                    dtype=torch.bool,
                    device=device,
                )
                policy = torch.tensor(
                    policy_data[i:i + batch_size],
                    dtype=torch.long,
                    device=device,
                )
                value = torch.tensor(
                    value_data[i:i + batch_size],
                    dtype=torch.long,
                    device=device,
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):
                    policy_predict, value_predict = (
                        transformer_net.forward_for_sl(
                            stones,
                            games,
                            stone_masks,
                        )
                    )

                    policy_loss = calculate_sl_policy_loss(
                        policy_predict,
                        policy,
                    )
                    value_loss = calculate_value_loss(
                        value_predict,
                        value,
                    )

                    loss = (
                        policy_loss + SL_VALUE_WEIGHT * value_loss
                    ).mean()

                test_loss["loss"] += loss.item()
                test_loss["policy"] += policy_loss.mean().item()
                test_loss["value"] += value_loss.mean().item()
                test_iteration += 1

    if test_iteration > 0:
        print_evaluation_information(
            test_loss,
            epochs,
            test_iteration,
            testing_time,
            data_name="Test",
        )
    else:
        print("Test : no test batch.")

    model_path = program_dir / "model" / f"{model_name}.bin"
    save_model(transformer_net, model_path)
    print("Finished Training.")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
