from nn.utility import plot_loss_history

if __name__ == "__main__":
    #file = input("input file name: ")
        
    file_path = "record/cai50000cp-32-9-LeaRate000.json"
        
    # ファイルから読み込んでプロット
    plot_loss_history(file_path)