from nn.utility import plot_loss_history
from pathlib import Path

if __name__ == "__main__":
    #file = input("input file name: ")
        
    file_path = str(Path(__file__).resolve().parent / "record" /
                    "transformer-sl-9-15-model-05-26-adamw-epoch50.json")
                    
    # ファイルから読み込んでプロット
    plot_loss_history(file_path)