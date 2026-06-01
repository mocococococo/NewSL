import click
from pathlib import Path

from dc3client import SocketClient
from dc3client.models import StoneRotation
from nn.utility import get_torch_device, load_network
from transformer.utility import load_transformer_network
from common.translate_state import convert_scores_to_dict, convert_stones_to_list, \
    scores_to_scorediff_for_team0, convert_team_stoi

from shot.search import set_root_state, shot_search


@click.command()
@click.option('--host', type=str, default="localhost", help='Host name (default: localhost)')
@click.option('--port', type=int, default=10000, help='Port number (default: 10000)')
@click.option('--model', type=str, default="Default.bin", help='Model name (default: sl-model.bin)')
@click.option('--transformer_model', type=str, default="Transformer.bin", help='Transformer model name (default: Transformer.bin)')
@click.option('--use_gpu', type=bool, default=True, help='use_gpu (default: True)')
@click.option('--name', type=str, default="SHOT_NewSL", help='AIname (default: SHOT_NewSL)')
@click.option('--debug', type=bool, default=False, help='debug (default: False)')
@click.option('--use_transformer', type=bool, default=False, help='use_transformer (default: False)')
@click.option('--transformer_target_end', type=int, multiple=True, default=(9, 10), help='transformer_target_end (default: 9). Can specify multiple values.')
@click.option('--transformer_target_shot', type=int, multiple=True, default=(15,), help='transformer_target_shot (default: 15). Can specify multiple values.')

def main(**kwargs):
    host = kwargs['host']
    port = kwargs['port']
    model = Path(Path(__file__).resolve().parents[0]) / "model" / kwargs['model']
    transformer_model = Path(Path(__file__).resolve().parents[0]) / "model" / kwargs['transformer_model']
    use_gpu = kwargs['use_gpu']
    cli_name = kwargs['name']
    debug = kwargs['debug']
    use_transformer = kwargs['use_transformer']
    transformer_target_end = kwargs['transformer_target_end']
    transformer_target_shot = kwargs['transformer_target_shot']

    cli = SocketClient(host=host, port=port, client_name=cli_name, auto_start=True, rate_limit=0.5)
    remove_trajectory = True

    my_team = cli.get_my_team()
    cli.logger.info(f"my_team :{my_team}")

    dc = cli.get_dc()
    dc_message = cli.convert_dc(dc)
    is_ready = cli.get_is_ready()

    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    transformer_network = None
    if use_transformer:
        if not transformer_model.exists():
            raise FileNotFoundError(f"transformer model not found: {transformer_model}")
        transformer_network = load_transformer_network(transformer_model, use_gpu=use_gpu)

    is_ready_message = cli.convert_is_ready(is_ready)

    while True:
        cli.update()
        match_data = cli.get_match_data()

        if (winner := cli.get_winner()) is not None:
            if my_team == winner:
                cli.logger.info("WIN")
            else:
                cli.logger.info("LOSE")
            break

        next_team = cli.get_next_team()

        if my_team == next_team:
            stones = convert_stones_to_list(match_data.update_list[-1].state.stones)
            scores = convert_scores_to_dict(match_data.update_list[-1].state.scores)
            end = match_data.update_list[-1].state.end
            shot = match_data.update_list[-1].state.shot
            hammer = convert_team_stoi(match_data.update_list[-1].state.hammer)
            score_diff_for_team0 = scores_to_scorediff_for_team0(scores)

            root_state = set_root_state(
                network=network,
                stones=stones,
                score_diff=score_diff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                transformer_network=transformer_network,
                debug=debug,
                use_transformer=use_transformer,
                transformer_target_end=transformer_target_end,
                transformer_target_shot=transformer_target_shot,
            )
            vx, vy, spin = shot_search(root_state, debug=debug)
            spin = StoneRotation.clockwise if spin == 0 else StoneRotation.counterclockwise

            cli.move(x=vx, y=vy, rotation=spin)
        else:
            continue

    move_info = cli.get_move_info()
    update_list, trajectory_list = cli.get_update_and_trajectory(remove_trajectory)


if __name__ == '__main__':
    main()
