import click
from typing import List

from dc3client import SocketClient
from dc3client.models import Stones
from nn.feature import discretization
from board.constant import X_MIN, X_MAX, Y_MIN, Y_MAX, STONE_RADIUS, Y_TEE, R_HOUSE, DCL2_YPOS_DIFF

import matplotlib.pyplot as plt

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
    

def show_sheet(stones) -> None:
    """シートを描画し、Enterが押されたらウィンドウを閉じて戻る"""
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_xlim(X_MIN - 0.5, X_MAX + 0.5)
    ax.set_ylim(Y_MIN - 0.5, Y_MAX + 0.5)
    ax.set_aspect('equal', adjustable='box')

    # ハウスの描画
    house_colors = ['#FF0000', '#FFFFFF', '#0000FF']  # 赤、白、青
    house_radii = [R_HOUSE, R_HOUSE * 2 / 3, R_HOUSE / 3]
    for radius, color in zip(house_radii, house_colors):
        ax.add_patch(plt.Circle((0.0, Y_TEE), radius, color=color, fill=True, alpha=0.5))

    # ストーンの描画
    for i, stone in enumerate(stones):
        if stone is None:
            continue
        x = stone["position"]["x"]
        y = stone["position"]["y"] - DCL2_YPOS_DIFF  # y座標を変換
        color = "red" if i < 8 else "blue"
        ax.add_patch(plt.Circle((x, y), STONE_RADIUS, color=color, fill=True))

    ax.set_title("Curling Sheet (Press Enter to continue)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True)

    done = {"flag": False}

    def on_key(event):
        if event.key in ("enter", "return"):
            done["flag"] = True

    cid = fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=False)

    # Enter が押されるまで待つ（ウィンドウを閉じた場合も抜ける）
    while plt.fignum_exists(fig.number) and not done["flag"]:
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)



@click.command()
@click.option('--host', type=str, default="localhost", help='Host name (default: localhost)')
@click.option('--port', type=int, default=10000, help='Port number (default: 10000)')
@click.option('--name', type=str, default="HumanClient", help='AIname (default: True)')

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
    cli_name = kwargs['name']

    # SocketClientには以下の引数を渡すことができます
    # host : デジタルカーリングを実行しているサーバーのIPアドレスを指定します。名前解決可能であればホスト名でも指定可能です。
    # port : デジタルカーリングを実行しているサーバーの指定されたポート番号を指定します。
    # client_name : クライアントの名前を指定します。デフォルトでは"AI0"となっています。
    # auto_start : サーバーに接続した際に自動で試合を開始するかどうかを指定します。デフォルトではTrueとなっています。
    # これは、dc3のコンバート機能のみを使用したいときにサーバーを起動する必要をなくすために用意されています。
    # rate_limit : 通信のレート制限を指定します。デフォルトでは2.0秒に1回となっています。早すぎるとサーバーから切断される可能性があります。
    cli = SocketClient(host=host, port=port, client_name=cli_name, auto_start=True, rate_limit=1.0)

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
            for s in stones:
                print(s)
            show_sheet(stones)

            selected_x = float(input("Enter selected x: "))
            selected_y = float(input("Enter selected y: "))
            rotation = input("Enter selected rotation (cw/ccw): ")
            selected_rotation = "cw" if rotation == "0" else "ccw"
    
            cli.move(x=selected_x, y=selected_y, rotation=selected_rotation)
        else:
            # 次のチームが自分のチームでなければ、何もしません
            continue

    # 試合が終了したら、clientから試合データを取得します
    move_info = cli.get_move_info()
    update_list, trajectory_list = cli.get_update_and_trajectory(remove_trajectory)


if __name__ == '__main__':
    main()
