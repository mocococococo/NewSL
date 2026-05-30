import click
from pathlib import Path

from dc3client import SocketClient
from dc3client.models import StoneRotation
from nn.utility import get_torch_device, load_network
from transformer.utility import load_transformer_network
from common.translate_state import (
    convert_scores_to_dict,
    convert_stones_to_list,
    convert_team_stoi,
    scores_to_scorediff_for_team0,
)

from shot.params import (
    DEFAULT_SHOT_INITIAL_CANDIDATES,
    DEFAULT_SHOT_MAX_DEPTH,
    DEFAULT_SHOT_MAX_SIMULATIONS,
    DEFAULT_SHOT_TIME_LIMIT_SEC,
)
from shot.search import set_root_state, shot_search


@click.command()
@click.option("--host", type=str, default="localhost", help="Host name (default: localhost)")
@click.option("--port", type=int, default=10000, help="Port number (default: 10000)")
@click.option("--model", type=str, default="Default.bin", help="Model name (default: Default.bin)")
@click.option(
    "--transformer_model",
    type=str,
    default="Transformer.bin",
    help="Transformer model name (default: Transformer.bin)",
)
@click.option("--use_gpu", type=bool, default=True, help="use_gpu (default: True)")
@click.option("--name", type=str, default="SHOT_NewSL", help="AI name (default: SHOT_NewSL)")
@click.option("--debug", type=bool, default=False, help="debug (default: False)")
@click.option("--use_transformer", type=bool, default=False, help="use_transformer (default: False)")
@click.option(
    "--transformer_target_end",
    type=int,
    multiple=True,
    default=(9, 10),
    help="Transformer target end. Can specify multiple values.",
)
@click.option(
    "--transformer_target_shot",
    type=int,
    multiple=True,
    default=(15,),
    help="Transformer target shot. Can specify multiple values.",
)
@click.option(
    "--shot_initial_candidates",
    type=int,
    default=DEFAULT_SHOT_INITIAL_CANDIDATES,
    help="Initial policy top-k candidates for SHOT.",
)
@click.option(
    "--shot_max_simulations",
    type=int,
    default=DEFAULT_SHOT_MAX_SIMULATIONS,
    help="Maximum SHOT simulations.",
)
@click.option(
    "--shot_time_limit_sec",
    type=float,
    default=DEFAULT_SHOT_TIME_LIMIT_SEC,
    help="SHOT time limit in seconds.",
)
@click.option(
    "--shot_max_depth",
    type=int,
    default=DEFAULT_SHOT_MAX_DEPTH,
    help="SHOT evaluation depth.",
)
def main(**kwargs):
    host = kwargs["host"]
    port = kwargs["port"]
    program_dir = Path(__file__).resolve().parent
    model = program_dir / "model" / kwargs["model"]
    transformer_model = program_dir / "model" / kwargs["transformer_model"]
    use_gpu = kwargs["use_gpu"]
    cli_name = kwargs["name"]
    debug = kwargs["debug"]
    use_transformer = kwargs["use_transformer"]
    transformer_target_end = kwargs["transformer_target_end"]
    transformer_target_shot = kwargs["transformer_target_shot"]
    shot_initial_candidates = kwargs["shot_initial_candidates"]
    shot_max_simulations = kwargs["shot_max_simulations"]
    shot_time_limit_sec = kwargs["shot_time_limit_sec"]
    shot_max_depth = kwargs["shot_max_depth"]

    cli = SocketClient(host=host, port=port, client_name=cli_name, auto_start=True, rate_limit=0.5)
    remove_trajectory = True

    my_team = cli.get_my_team()
    cli.logger.info(f"my_team :{my_team}")

    cli.get_dc()
    cli.get_is_ready()

    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    transformer_network = None
    if use_transformer:
        if not transformer_model.exists():
            raise FileNotFoundError(f"transformer model not found: {transformer_model}")
        transformer_network = load_transformer_network(transformer_model, use_gpu=use_gpu)

    while True:
        cli.update()
        match_data = cli.get_match_data()

        if (winner := cli.get_winner()) is not None:
            cli.logger.info("WIN" if my_team == winner else "LOSE")
            break

        next_team = cli.get_next_team()
        if my_team != next_team:
            continue

        state = match_data.update_list[-1].state
        stones = convert_stones_to_list(state.stones)
        scores = convert_scores_to_dict(state.scores)
        end = state.end
        shot = state.shot
        hammer = convert_team_stoi(state.hammer)
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
        vx, vy, spin = shot_search(
            root_state,
            initial_candidates=shot_initial_candidates,
            max_simulations=shot_max_simulations,
            time_limit_sec=shot_time_limit_sec,
            max_depth=shot_max_depth,
            debug=debug,
        )
        spin = StoneRotation.clockwise if spin == 0 else StoneRotation.counterclockwise
        cli.move(x=vx, y=vy, rotation=spin)

    cli.get_move_info()
    cli.get_update_and_trajectory(remove_trajectory)


if __name__ == "__main__":
    main()
