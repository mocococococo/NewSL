from pathlib import Path
import click
from learning_param import BATCH_SIZE, EPOCHS
from board.constant import BOARD_SIZE
from nn.learn import train_on_cpu, train_on_gpu
from nn.data_generator import generate_supervised_learning_data

@click.command()
@click.option('--model-name', type=click.STRING, default="sl-model.bin", help="保存するモデルの名前の指定")
@click.option('--use-gpu', type=click.BOOL, default=True, help="GPUの使用")
def train_main(model_name: str, use_gpu: bool):
    # プログラムのディレクトリ
    program_dir = Path(__file__).resolve().parent
    
    # 対戦データのlogファイルがあるディレクトリ
    log_dir = "path to data directory."
    size = BOARD_SIZE
    
    print(f"start learning model {model_name} !!")
    
    generate_supervised_learning_data(program_dir, log_dir, data_size=100)
    #return
    if use_gpu:
        print("use gpu")
        train_on_gpu(program_dir=program_dir,board_size=size, batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=model_name)
    else :
        print("use cpu")
        train_on_cpu(program_dir=program_dir,board_size=size, batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=model_name)
    print(f"finish learning model {model_name} !!")

if __name__ == "__main__":
    train_main()
