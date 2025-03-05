import click
from typing import List

from dc3client import SocketClient
from dc3client.models import Stones
from nn.feature import generate_input_planes
from nn.utility import get_torch_device, load_network
from policy_shot import generate_move_from_policy


def convert_stones_to_list(stones: Stones) -> List[dict]:
    result = [None] * 16  # 16要素のリストを作成し、全てをNoneで初期化
    for i, coordinate in enumerate(stones.team0):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {"x": coordinate.position[0].x, "y": coordinate.position[0].y}
            }
            result[i] = data

    for i, coordinate in enumerate(stones.team1):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {"x": coordinate.position[0].x, "y": coordinate.position[0].y}
            }
            result[i + 8] = data

    return result
    



@click.command()
@click.option('--host', type=str, default="localhost", help='Host name (default: localhost)')
@click.option('--port', type=int, default=10000, help='Port number (default: 10000)')
@click.option('--model', type=str, default="Default.bin", help='Model name (default: sl-model.bin)')
@click.option('--use_gpu', type=bool, default=True, help='use_gpu (default: True)')
@click.option('--name', type=str, default="NewSL", help='AIname (default: True)')

def main(**kwargs):
    # 機械学習のモデルなど、時間のかかる処理はここで行います。
    # 通信プロトコルの解説において、is_readyを受け取ってからreadyを返すまでに行うことを推奨しています。
    # しかしながら、そのタイミングでエラーが生じるとサーバー自体の動作が停止してしまうため、すべての準備が終わってから接続を行うことを推奨します。
    # dc3の推奨に従う場合、以下のようになります。
    # 1. auto_startをFalseにしてSocketClientを初期化する
    #   cli = SocketClient(auto_start=False)
    # 2. サーバーに接続する
    #   cli.connect(cli.server)
    # 3. dcを受け取る
    #   cli.dc_receive()
    # 4. dc_okを送信する
    #   cli.dc_ok()
    # 5. is_readyを受け取る
    #   cli.is_ready_recv()
    # 6. モデルを読み込むなどの準備を行う
    # 7. ready_okを送信する
    #   cli.ready_ok()
    # 8. サーバーからの開始指示を待つ
    #   cli.get_new_game()
    host = kwargs['host']
    port = kwargs['port']
    model = "./model/" + kwargs['model']
    use_gpu = kwargs['use_gpu']
    cli_name = kwargs['name']

    # SocketClientには以下の引数を渡すことができます
    # host : デジタルカーリングを実行しているサーバーのIPアドレスを指定します。名前解決可能であればホスト名でも指定可能です。
    # port : デジタルカーリングを実行しているサーバーの指定されたポート番号を指定します。
    # client_name : クライアントの名前を指定します。デフォルトでは"AI0"となっています。
    # auto_start : サーバーに接続した際に自動で試合を開始するかどうかを指定します。デフォルトではTrueとなっています。
    # これは、dc3のコンバート機能のみを使用したいときにサーバーを起動する必要をなくすために用意されています。
    # rate_limit : 通信のレート制限を指定します。デフォルトでは2.0秒に1回となっています。早すぎるとサーバーから切断される可能性があります。
    cli = SocketClient(host=host, port=port, client_name=cli_name, auto_start=True, rate_limit=0.5)

    # ログを出力するディレクトリを指定します。デフォルトでは"logs/"となっています。
    #log_dir = pathlib.Path("logs")

    # データ保存時に、軌跡データを削除するかどうかを指定します。デフォルトではTrueとなっています。
    # 軌跡データを保存すると容量が膨大になるため、必要ない場合はTrueにしてください。
    remove_trajectory = True

    # 自分がteam0かteam1かを取得します
    my_team = cli.get_my_team()
    cli.logger.info(f"my_team :{my_team}")

    # dcやis_readyをdataclass形式で取得し、保存しやすいようにdict形式に変換します。
    dc = cli.get_dc()
    dc_message = cli.convert_dc(dc)
    is_ready = cli.get_is_ready()

    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    is_ready_message = cli.convert_is_ready(is_ready)

    # 試合を開始します
    while True:

        # updateを受け取ります
        cli.update()

        # 試合状況を取得します
        # 現在の情報は、match_data.update_listに順番に格納されています
        match_data = cli.get_match_data()
        #print("match_data :", match_data.update_list[-1].state.stones.team0)
        
        

        # winnerが存在するかどうかで、試合が終了しているかどうかを確認します
        if (winner := cli.get_winner()) is not None:
            # game end
            if my_team == winner:
                # 勝利
                cli.logger.info("WIN")
            else:
                # 敗北
                cli.logger.info("LOSE")
            # 試合が終了したらループを抜けます
            break

        # 次のチームが自分のチームかどうかを確認します
        next_team = cli.get_next_team()

        # 次のチームが自分のチームであれば、moveを送信します
        if my_team == next_team:
            # 実際の投球は、move関数を呼び出すことで行います
            # move関数の引数は、x, y, rotationの3つです
            # x, yはそれぞれ投球する石のx(横)方向成分、y(縦)方向成分を指定します
            # rotationは投球する石の回転方向を指定します
            # このとき、rotationにはStoneRotationクラスの値を指定します
            # StoneRotation.clockwise : 時計回り
            # StoneRotation.inturn : インターン = 時計回り
            # StoneRotation.counterclockwise : 反時計回り
            # StoneRotation.outturn : アウトターン = 反時計回り
            stones = convert_stones_to_list(match_data.update_list[-1].state.stones)
            inputplanes = generate_input_planes(stones, match_data.update_list[-1].state.shot)
            shot_index = match_data.update_list[-1].state.shot

            selected_x, selected_y, selected_rotation = generate_move_from_policy(network, inputplanes, shot_index)
    
            cli.move(x=selected_x, y=selected_y, rotation=selected_rotation)
        else:
            # 次のチームが自分のチームでなければ、何もしません
            continue

    # 試合が終了したら、clientから試合データを取得します
    move_info = cli.get_move_info()
    update_list, trajectory_list = cli.get_update_and_trajectory(remove_trajectory)

    '''# 試合データを保存します、
    update_dict = {}

    for update in update_list:
        # updateをdict形式に変換します
        update_dict = cli.convert_update(update, remove_trajectory)

    # updateを保存します、どのように保存するかは任意です
    with open("data.json", "w", encoding="UTF-8") as f:
        json.dump(update_dict, f, indent=4)'''


if __name__ == '__main__':
    main()
