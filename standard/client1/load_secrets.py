import os
from pathlib import Path
from dotenv import load_dotenv

# AI作成者一人一人に固有のユーザネーム、パスワードを設定して頂きます。
load_dotenv(Path(__file__).resolve().with_name(".env"))

username = os.getenv("MATCH_USER_NAME")
password = os.getenv("PASS_WORD")

if __name__ == "__main__":
    print(f"Username: {username}")
    print(f"Password: {password}")