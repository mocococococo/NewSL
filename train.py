import glob
import os
import click
from pathlib import Path
from learning_param import BATCH_SIZE, EPOCHS
from nn.learn import train_on_cpu, train_on_gpu
from nn.generator import generate_supervised_learning_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@click.command()
@click.option('--model-name', type=click.STRING, default="cai70000CP-32-9-LeaRate100-vx32-vy25-batchsize512", help="保存するモデルの名前の指定")
@click.option('--use-gpu', type=click.BOOL, default=True, help="GPUの使用")
def train_main(model_name: str, use_gpu: bool):
    # プログラムのディレクトリ
    program_dir = str(Path(__file__).resolve().parent)
    # 対戦データのlogファイルがあるディレクトリ
    log_dir = str(Path(__file__).resolve().parent / "LearnLog" / "cai")
    
    print(f"start learning model {model_name} !!")
    
    generate_supervised_learning_data(program_dir, log_dir, data_size=80000)
    # return
    if use_gpu:
        train_on_gpu(program_dir=program_dir, batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=model_name)
    else :
        train_on_cpu(program_dir=program_dir, batch_size=BATCH_SIZE, epochs=EPOCHS, model_name=model_name)
    print(f"finish learning model {model_name} !!")

if __name__ == "__main__":
    train_main()