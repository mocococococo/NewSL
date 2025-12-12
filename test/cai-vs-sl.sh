# config.jsonの相対パス
json_file="config.json"

# 新しいポート番号
port0=10000
port1=10001

# N 回自己対戦させる
for i in {1..100}; do
    echo "Round $i"
    # バックグラウンドでサーバーを立ち上げる
    ./server.exe &
    sleep 2

    # バックグラウンドでクライアントを立ち上げる
    ./sl-player.sh $port0 &
    ./cai.sh $port1 &

    wait

    # swap=$((port0))
    # port0=$((port1))
    # port1=$((swap))
done