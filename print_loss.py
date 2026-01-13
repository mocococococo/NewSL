from nn.utility import plot_loss_history
from pathlib import Path

if __name__ == "__main__":
    #file = input("input file name: ")
        
    file_path = str(Path(__file__).resolve().parent / "record" /
                    "cai70000CP-32-9-LeaRate0625-vx32-vy25-batchsize512.json")
                    
    # ファイルから読み込んでプロット
    plot_loss_history(file_path)