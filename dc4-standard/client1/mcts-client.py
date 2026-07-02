import asyncio
import json
import click
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../NewSL
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_secrets import username, password
from dc4client.dc_client import DCClient
from dc4client.send_data import TeamModel, MatchNameModel
from dc3client.models import StoneRotation
from nn.utility import get_torch_device, load_network
from common.translate_state import convert_dc4_stones_to_list, convert_dc4_scores_to_dict, \
    scores_to_scorediff_for_team0, get_hammer_team
from mcts.search import set_root_state, mcts_search

formatter = logging.Formatter(
    "%(asctime)s, %(name)s : %(levelname)s - %(message)s"
)

global_end = -1
global_shot = -1

@click.command()
@click.option('--host', type=str, default="localhost", help='Host name (default: localhost)')
@click.option('--port', type=int, default=5000, help='Port number (default: 10000)')
@click.option('--model', type=str, default="Default.bin", help='Model name (default: sl-model.bin)')
@click.option('--use_gpu', type=bool, default=True, help='use_gpu (default: True)')
@click.option('--debug', type=bool, default=False, help='debug (default: False)')

def main(**kwargs):
    asyncio.run(run_client(**kwargs))
    
async def run_client(**kwargs):
    # 引数の読み込み
    host = kwargs['host']
    port = kwargs['port']
    model = "./model/" + kwargs['model']
    use_gpu = kwargs['use_gpu']
    debug = kwargs['debug']
    
    # match_idの読み込みます。
    json_path = Path(__file__).parents[1] / "match_id.json"
    with open(json_path, "r") as f:
        match_id = json.load(f)

    # 最初のエンドにおいて、team0が先攻、team1が後攻です。
    # デフォルトではmatch_team_name=team1となっており、先攻に切り替えたい場合はDCClientのコンストラクタの引数にて
    # match_team_name=MatchNameModel.team0
    # としてください
    # クライアントの初期化（ログレベルはデフォルトでINFO、保存機能はデフォルトでTrue）
    client = DCClient(match_id=match_id, username=username, password=password, match_team_name=MatchNameModel.team0, auto_save_log=False, log_dir="logs")

    # ここで、接続先のサーバのアドレスとポートを指定します。
    # デフォルトではlocalhost:5000となっています。
    # こちらは接続先に応じて変更してください。
    client.set_server_address(host=host, port=port)

    # チーム設定の読み込み
    with open(Path(__file__).resolve().parent / "team_config.json", "r") as f:
        data = json.load(f)
    client_data = TeamModel(**data)

    # ログ設定(不要であれば削除してください)
    # DCClient内にもloggerがあるため、そちらを利用することも可能ですが、
    # client.logger を使用するとライブラリ側で管理しているバッファに自動的に入ります
    logger = logging.getLogger("MCTS_NewSL")
    logger.setLevel(level=logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"client_data.team_name: {client_data.team_name}")
    logger.debug(f"client_data: {client_data}")
    
    # 機械学習のモデルなど、時間のかかる処理はここで行います。
    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    # チーム情報をサーバに送信します。
    # 相手のクライアントも同様にチーム情報を送信するまで待機します。
    # 送信後、自チームの名前を受け取ります(team0 または team1)。
    # 両チームが揃うと試合が開始され、思考時間のカウントが始まります。
    # そのため、AIの初期化などはこの前に行ってください。
    match_team_name: MatchNameModel = await client.send_team_info(client_data)

    try:
        async for state_data in client.receive_state_data():
            
            # ゲーム終了の判定
            if (winner_team := client.get_winner_team()) is not None:
                logger.info(f"Winner: {winner_team}")
                break
            
            next_shot_team = client.get_next_team()
            score = state_data.score
            logger.info(
                f"end={state_data.end_number}, shot={state_data.shot_number}, total={state_data.total_shot_number}, "
                f"next={state_data.next_shot_team}, score_t0={sum(score.team0) if score else None}, score_t1={sum(score.team1) if score else None}"
            )
            
            if is_finished(state_data.end_number, state_data.total_shot_number):
                print("既に処理済みのデータのためスキップします。")
                continue

            # AIを実装する際の処理はこちらになります。
            if next_shot_team == match_team_name:
                stones = convert_dc4_stones_to_list(state_data.stone_coordinate.data)
                scores = convert_dc4_scores_to_dict(state_data.score)
                end = client.get_end_number()
                shot = client.get_shot_number()
                hammer = get_hammer_team(match_team_name, shot)
                score_diff_for_team0 = scores_to_scorediff_for_team0(scores) if scores else 0
                
                root_state = set_root_state(
                    network=network,
                    stones=stones,
                    score_diff=score_diff_for_team0,
                    end=end,
                    shot_index=shot,
                    hammer_team=hammer,
                    debug=debug
                )
                vx, vy, spin = mcts_search(root_state, debug=debug)
                spin = StoneRotation.clockwise if spin == 0 else StoneRotation.counterclockwise

                await client.send_shot_info_dc3(
                    vx=vx,
                    vy=vy,
                    rotation=spin
                )

    except Exception as e:
        client.logger.error(f"Unexpected error in main loop: {e}")
    
    finally:
        # 試合終了後、あるいはエラー時に溜まったログをファイルに書き出す
        # ファイル名（チーム名や時刻）の生成やディレクトリ作成はライブラリが自動で行います
        client.save_log_file()

def is_finished(end: int, shot: int) -> bool:
    # DCClient から非同期で受け取ったデータが既に処理済みのものかどうかを判定する関数
    if end is None or shot is None:
        return False
    global global_end, global_shot
    if end < global_end or (end == global_end and shot <= global_shot):
        return True
    elif end > global_end or (end == global_end and shot > global_shot):
        global_end = end
        global_shot = shot
        return False
    

if __name__ == "__main__":
    main()