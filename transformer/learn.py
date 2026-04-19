"""Transformer ネットワークの教師あり学習。"""

from __future__ import annotations

import glob
import time
from pathlib import Path

import torch

from learning_param import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_SCHEDULE,
    MOMENTUM,
    SL_LEARNING_RATE,
    SL_VALUE_WEIGHT,
    WEIGHT_DECAY,
)
from transformer.loss import calculate_kld_loss
from transformer.network import TransformerNetwork
from transformer.utility import (
    get_torch_device,
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

    optimizer = torch.optim.SGD(
        transformer_net.parameters(),
        lr=SL_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    current_lr = SL_LEARNING_RATE

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

                    loss = policy_loss + SL_VALUE_WEIGHT * value_loss

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

                        loss = policy_loss + SL_VALUE_WEIGHT * value_loss

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

        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    model_path = program_dir / "model" / f"{model_name}.bin"
    save_model(transformer_net, model_path)
    print("Finished Training.")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train()
