import click
from datetime import datetime
from pathlib import Path

from dc3client import SocketClient
from dc3client.models import StoneRotation
from nn.utility import get_torch_device, load_network
from transformer.params import TRANSFORMER_VY_MODE
from transformer.utility import load_transformer_network
from common.translate_state import convert_scores_to_dict, convert_stones_to_list, \
    scores_to_scorediff_for_team0, convert_team_stoi

from mcts.search import set_root_state, mcts_search


DEFAULT_TRANSFORMER_MODELS_BY_SHOT = {
    4: "transformer-sl-9-4-model-06-28-adamw-epoch50-shot.bin",
    5: "transformer-sl-9-5-model-06-27-adamw-epoch50-shot.bin",
    6: "transformer-sl-9-6-model-06-19-adamw-epoch50-shot.bin",
    7: "transformer-sl-9-7-model-06-18-adamw-epoch50-shot.bin",
    8: "transformer-sl-9-8-model-06-16-adamw-epoch50-shot.bin",
    9: "transformer-sl-9-9-model-06-14-adamw-epoch50-shot.bin",
    10: "transformer-sl-9-10-model-06-11-adamw-epoch50-shot.bin",
    11: "transformer-sl-9-11-model-06-09-adamw-epoch50-shot.bin",
    12: "transformer-sl-9-12-model-06-08-adamw-epoch50-shot.bin",
    13: "transformer-sl-9-13-model-06-06-adamw-epoch50-shot.bin",
    14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
    15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
}


def _model_path(model_name: str) -> Path:
    return Path(__file__).resolve().parent / "model" / model_name


def _load_transformer_networks_by_shot(
    transformer_model: str | None,
    transformer_models_by_shot: dict[int, str],
    transformer_target_shot: tuple[int, ...],
    use_gpu: bool,
    action_type,
):
    networks_by_shot = {}
    for shot in sorted(set(int(shot) for shot in transformer_target_shot)):
        model_name = transformer_models_by_shot.get(shot, transformer_model)
        if model_name is None:
            raise ValueError(f"transformer model for shot {shot} is not configured")

        model_path = _model_path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"transformer model for shot {shot} not found: {model_path}")
        networks_by_shot[shot] = load_transformer_network(
            model_path,
            use_gpu=use_gpu,
            action_type=action_type,
        )
    return networks_by_shot


def resolve_stats_log_path(stats_log_path):
    if not stats_log_path:
        return None

    path = Path(stats_log_path)
    if path.exists() and path.is_dir():
        return path / f"puct_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    if not path.exists() and path.suffix == "":
        return path / f"puct_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return path


@click.command()
@click.option('--host', type=str, default="localhost", help='Host name (default: localhost)')
@click.option('--port', type=int, default=10000, help='Port number (default: 10000)')
@click.option('--sl_model', type=str, default="Default.bin", help='教師あり学習モデル名 (default: Default.bin)')
@click.option('--sl_model_is_cnn', type=bool, default=True, help='教師あり学習モデルがCNNかどうか (default: True)')
@click.option('--transformer_model', type=str, default=None, help='Single Transformer model name used as fallback')
@click.option('--transformer_model_4', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[4], help='Transformer model name for shot 4')
@click.option('--transformer_model_5', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[5], help='Transformer model name for shot 5')
@click.option('--transformer_model_6', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[6], help='Transformer model name for shot 6')
@click.option('--transformer_model_7', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[7], help='Transformer model name for shot 7')
@click.option('--transformer_model_8', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[8], help='Transformer model name for shot 8')
@click.option('--transformer_model_9', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[9], help='Transformer model name for shot 9')
@click.option('--transformer_model_10', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[10], help='Transformer model name for shot 10')
@click.option('--transformer_model_11', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[11], help='Transformer model name for shot 11')
@click.option('--transformer_model_12', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[12], help='Transformer model name for shot 12')
@click.option('--transformer_model_13', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[13], help='Transformer model name for shot 13')
@click.option('--transformer_model_14', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[14], help='Transformer model name for shot 14')
@click.option('--transformer_model_15', type=str, default=DEFAULT_TRANSFORMER_MODELS_BY_SHOT[15], help='Transformer model name for shot 15')
@click.option('--use_gpu', type=bool, default=True, help='use_gpu (default: True)')
@click.option('--name', type=str, default="MCTS_NewSL", help='AIname (default: True)')
@click.option('--debug', type=bool, default=False, help='debug (default: False)')
@click.option('--stats_log_path', type=str, default=None, help='stats_log_path (default: None)')
@click.option('--use_search_based_model', type=bool, default=False, help='探索統計学習モデルを使用するかどうか (default: False)')
@click.option('--transformer_target_end', type=int, multiple=True, default=(9, 10), help='transformer_target_end (default: 9). Can specify multiple values.')
@click.option('--transformer_target_shot', type=int, multiple=True, default=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,), help='transformer_target_shot (default: 15). Can specify multiple values.')
@click.option('--use_progressive_widening', type=bool, default=True, help='use Progressive Widening (default: True)')
@click.option('--use_transposition_table', type=bool, default=True, help='use Transposition Table (default: True)')
@click.option('--measure_tt_stats', type=bool, default=False, help='TT統計を計測するかどうか (default: True)')

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
    sl_model_path = _model_path(kwargs['sl_model'])
    transformer_model = kwargs['transformer_model']
    transformer_models_by_shot = {
        4: kwargs['transformer_model_4'],
        5: kwargs['transformer_model_5'],
        6: kwargs['transformer_model_6'],
        7: kwargs['transformer_model_7'],
        8: kwargs['transformer_model_8'],
        9: kwargs['transformer_model_9'],
        10: kwargs['transformer_model_10'],
        11: kwargs['transformer_model_11'],
        12: kwargs['transformer_model_12'],
        13: kwargs['transformer_model_13'],
        14: kwargs['transformer_model_14'],
        15: kwargs['transformer_model_15'],
    }
    sl_model_is_cnn = kwargs['sl_model_is_cnn']
    use_gpu = kwargs['use_gpu']
    cli_name = kwargs['name']
    debug = kwargs['debug']
    stats_log_path = resolve_stats_log_path(kwargs['stats_log_path'])
    use_search_based_model = kwargs['use_search_based_model']
    transformer_target_end = kwargs['transformer_target_end']
    transformer_target_shot = kwargs['transformer_target_shot']
    use_progressive_widening = kwargs['use_progressive_widening']
    use_transposition_table = kwargs['use_transposition_table']
    measure_tt_stats = kwargs['measure_tt_stats']

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

    action_type = "default" if sl_model_is_cnn else TRANSFORMER_VY_MODE
    if sl_model_is_cnn:
        device = get_torch_device(use_gpu=use_gpu)
        sl_model = load_network(sl_model_path, use_gpu=use_gpu)
        sl_model.to(device)
    else:
        sl_model = load_transformer_network(
            sl_model_path,
            use_gpu=use_gpu,
            action_type=action_type,
        )

    search_based_model = None
    if use_search_based_model:
        search_based_model = _load_transformer_networks_by_shot(
            transformer_model,
            transformer_models_by_shot,
            transformer_target_shot,
            use_gpu,
            action_type,
        )
        
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
            scores = convert_scores_to_dict(match_data.update_list[-1].state.scores)
            end = match_data.update_list[-1].state.end
            shot = match_data.update_list[-1].state.shot
            hammer = convert_team_stoi(match_data.update_list[-1].state.hammer)
            score_diff_for_team0 = scores_to_scorediff_for_team0(scores)

            root_state = set_root_state(
                sl_model=sl_model,
                stones=stones,
                score_diff=score_diff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                debug=debug,
                sl_model_is_cnn=sl_model_is_cnn,
                search_based_model=search_based_model,
                use_search_based_model=use_search_based_model,
                transformer_target_end=transformer_target_end,
                transformer_target_shot=transformer_target_shot,
            )
            vx, vy, spin = mcts_search(
                root_state,
                debug=debug,
                stats_log_path=stats_log_path,
                use_progressive_widening=use_progressive_widening,
                use_transposition_table=use_transposition_table,
                measure_tt_stats=measure_tt_stats,
                action_type=action_type,
            )
            spin = StoneRotation.clockwise if spin == 0 else StoneRotation.counterclockwise

            cli.move(x=vx, y=vy, rotation=spin)
        else:
            # 次のチームが自分のチームでなければ、何もしません
            continue

    # 試合が終了したら、clientから試合データを取得します
    move_info = cli.get_move_info()
    update_list, trajectory_list = cli.get_update_and_trajectory(remove_trajectory)


if __name__ == '__main__':
    main()
