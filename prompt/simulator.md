カーリングの戦術をモンテカルロ木探索で考える際に、ショットのベクトルを選ぶ部分がショットを選んだあとに、このシミュレータを使って実際に局面をシミュレーションするには、どれに何を与えて呼び出せばいい？

つまり、pythonなら
- 16個のストーンの座標
- 何投目かのshot_index
- ショット{
    x座標,
    y座標,
    回転方向,
}
- フリーガードゾーンルールの有無
- 番外にストーンを残したいか
を与えて呼び出せばいい？

分かりました。
x, y座標は、何を基準にしているかは書かれていますか？



できました。
添付したpythonのファイルを実行したのですが、各行で呼ばれているAPIが何をするAPIなのかをそれぞれ説明してもらえますか？
- fs.shot2dest((-0.132, 2.3995, 0))
- fs.dest2shot((0, 38.405), 0)
- fs.passpoint2shot((0, 38.405), 3.5, 0)
- fs.passpointgo2shot((0, 38.405), 2.0, 1)
- fs.simulate([], 15, (-0.132, 2.3995, 0))
- fs.passpoint2shot(stones[0], 3.5, 1)
- fs.simulate(stones, 1, shot, freeguard=False)
- fs.simulate(stones, 1, shot, freeguard=False, rink_only=False)
- fs.passpoint2shot(stones[0], 3.5, 1)
- fs.simulate(stones, 1, shot)
- fs.simulate(stones, 1, shot, freeguard=True)

いくつか質問です。
fs.shot2dest((-0.132, 2.3995, 0)) について、引数となるショットの構成は
- x 方向のベクトル
- y 方向のベクトル
- 回転方向（0 または 1 で表される）
という認識でよいですか？

fs.dest2shot((0, 38.405), 0) について、引数の spin は目標の到達位置のためのショットの回転方向だけは指定するということですか？

fs.simulate([], 15, (-0.132, 2.3995, 0)) について、第4引数と第5引数はデフォルトがあって、引数を取るかは任意の形ですか？

- fs.passpoint2shot
- fs.passpointgo2shot
のユースケースの違いが分かりません。
どちらの引数にも同じものを指定すれば、同じショットが返ってきませんか？