"""深層学習の実装。
"""
from typing import NoReturn
import glob
import os
import time
import torch
from nn.network.dual_net import DualNet
from nn.loss import calculate_sl_policy_loss, calculate_value_loss
from nn.utility import get_torch_device, print_learning_process, \
    print_evaluation_information, save_model, load_data_set, \
    split_train_test_set, print_learning_result

from learning_param import SL_LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, SL_VALUE_WEIGHT, LEARNING_SCHEDULE

def train_on_cpu(program_dir: str, board_size: int, batch_size: int, \
    epochs: int, model_name: str="sl-model.bin") -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """
    # 学習データと検証用データの分割
    print(os.path.join(program_dir, "data", "sl_data_*.npz"))
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))
    print("success to get data_set.")
    print(data_set)
    train_data_set, test_data_set = split_train_test_set(data_set, 0.9)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=False)

    dual_net = DualNet(device=device)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    #scaler = torch.cuda.amp.GradScaler() #混合制度学習： 32bit浮動小数点数と16bit浮動小数点数をいい感じに切り替えて、結果の精度を保ったまま学習の高速化、メモリの節約ができる。

    current_lr = SL_LEARNING_RATE
    for epoch in range(epochs):
        for data_index, train_data_path in enumerate(train_data_set): #enumerate()：要素の内容と要素番号を取得するらしい
            plane_data, policy_data, value_data = load_data_set(train_data_path)
            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            dual_net.train() #nn.Moduleを継承しているDualNetクラスの関数でトレーニングモードなるものに設定するらしい
            epoch_time = time.time()
            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                value = torch.tensor(value_data[i:i+batch_size]).to(device)

                policy_predict, value_predict = dual_net.forward_with_softmax(plane) #ここで入力をネットワークに通して予測値を得ている。
                                    
                print("value_predict: ", value_predict.size())
                print("value: ", value.size())
                
                policy_loss = calculate_sl_policy_loss(policy_predict, policy) #sl抜きのものから変更。本来こっちがあっているはずだが、真偽不明。
                value_loss = calculate_value_loss(value_predict, value)
                print("policy_loss: ", policy_loss.size())
                print("value_loss: ", value_loss.size())

                dual_net.zero_grad() #すべてのパラメータの勾配を0に設定する。
                    

                loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean() #ポリシーの損失にバリューの損失（重み付き）を加算したベクトルの平均を損失としている。

                print("loss: ", loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #スケーリング：装置やソフトウェア、システムなどの性能や処理能力を、要求される処理量に合わせて増強したり縮減したりすること
                #scaler.scale(loss).backward() #lostがスケーリングファクタ？で乗算することでスケーリングし、逆伝播を行っている。
                #scaler.step(optimizer) #勾配が無限または非数出ない場合、optimizer.step()つまり重みの計算を行う。
                #scaler.update() #次の繰り返しに向けてスケールをアップデート。

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1

            print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_data_path in enumerate(test_data_set): #テストしてる。ネットワークの更新等は行っていない。
            dual_net.eval()
            plane_data, policy_data, value_data = load_data_set(test_data_path)
            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size], dtype=torch.long).to(device)
                    value = torch.tensor(value_data[i:i+batch_size], dtype=torch.long).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    policy_loss = calculate_sl_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1

        print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        if epoch in LEARNING_SCHEDULE["learning_rate"]: #特定のエポックの回数（現在は5,8,10）の時、学習率を変更
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    save_model(dual_net, os.path.join("model", f"{model_name}"))

def train_on_gpu(program_dir: str, board_size: int, batch_size: int, \
    epochs: int, model_name: str="sl-model.bin") -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """
    # 学習データと検証用データの分割
    print(os.path.join(program_dir, "data", "sl_data_*.npz"))
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))
    print("success to get data_set.")
    print(data_set)
    train_data_set, test_data_set = split_train_test_set(data_set, 0.9)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=True)

    dual_net = DualNet(device=device)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler() #混合制度学習： 32bit浮動小数点数と16bit浮動小数点数をいい感じに切り替えて、結果の精度を保ったまま学習の高速化、メモリの節約ができる。

    current_lr = SL_LEARNING_RATE
    
    loss_history = {
        "loss": [],
        "policy": [],
        "value": [],
    }

    for epoch in range(epochs):
        for data_index, train_data_path in enumerate(train_data_set): #enumerate()：要素の内容と要素番号を取得するらしい
            plane_data, policy_data, value_data = load_data_set(train_data_path)
            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            dual_net.train() #nn.Moduleを継承しているDualNetクラスの関数でトレーニングモードなるものに設定するらしい
            epoch_time = time.time()
            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size], dtype=torch.long).to(device)
                    value = torch.tensor(value_data[i:i+batch_size], dtype=torch.long).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    dual_net.zero_grad() #すべてのパラメータの勾配を0に設定する。

                    policy_loss = calculate_sl_policy_loss(policy_predict, policy) #sl抜きのものから変更。本来こっちがあっているはずだが、真偽不明。
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean() #ポリシーの損失にバリューの損失（重み付き）を加算したベクトルの平均を損失としている。

                #スケーリング：装置やソフトウェア、システムなどの性能や処理能力を、要求される処理量に合わせて増強したり縮減したりすること
                scaler.scale(loss).backward() #lostがスケーリングファクタ？で乗算することでスケーリングし、逆伝播を行っている。
                scaler.step(optimizer) #勾配が無限または非数出ない場合、optimizer.step()つまり重みの計算を行う。
                scaler.update() #次の繰り返しに向けてスケールをアップデート。

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1

            print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_data_path in enumerate(test_data_set): #テストしてる。ネットワークの更新等は行っていない。
            dual_net.eval()
            plane_data, policy_data, value_data = load_data_set(test_data_path)
            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size], dtype=torch.long).to(device)
                    value = torch.tensor(value_data[i:i+batch_size], dtype=torch.long).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    policy_loss = calculate_sl_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1
        
                
        loss_history["loss"].append(test_loss["loss"] / test_iteration)
        loss_history["policy"].append(test_loss["policy"] / test_iteration)
        loss_history["value"].append(test_loss["value"] / test_iteration)

        print_evaluation_information(test_loss, epoch, test_iteration, testing_time)


        if epoch in LEARNING_SCHEDULE["learning_rate"]: #特定のエポックの回数（現在は5,8,10）の時、学習率を変更
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    print_learning_result(loss_history)
    
    save_model(dual_net, os.path.join("model", f"{model_name}"))

