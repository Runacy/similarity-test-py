import cv2
import os

# お絵描きなのでスコア指標は特徴量抽出でいく
TARGET_FILE = 'rice2.jpeg'
IMG_DIR = "Image"
IMG_SIZE = (200, 200)


def calculateScore(path: str) -> tuple:
    target_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # グレースケール変換

    target_img = cv2.resize(target_img, IMG_SIZE)
    """
    総当たりマッチングの基礎
    cv2.BFMatcher *** 特徴点のマッチング用オブジェクトの作成かな

    cv2.NORM_HAMMING ***
    特徴ベクトル間のハミング距離の計算
    """

    detector = cv2.ORB_create()

    (target_kp, target_des) = detector.detectAndCompute(target_img, None)
    return (target_kp, target_des)


if __name__ == "__main__":
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    target_img_path = os.path.join(os.getcwd(), IMG_DIR, TARGET_FILE)

    (target_kp, target_des) = calculateScore(target_img_path)  # 計算
    print('Target_File: %s' % (TARGET_FILE))

    files = os.listdir(IMG_DIR)
    scores = []
    for file in files:
        if file == '.DS_Store' or file == TARGET_FILE or file == '.gitkeep':
            continue

        comparing_img_path = os.path.join(os.getcwd(), IMG_DIR, file)
        try:
            (comparing_kp, comparing_des) = calculateScore(
                comparing_img_path)  # 計算

            matches = bf.match(target_des, comparing_des)  # 総当たりで特徴点の距離を求める

            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)  # 特徴点の距離の平均を類似度のスコアとする
        except cv2.error as e:
            print(e)
            ret = 100000

        scores.append((file, ret))

    scores = sorted(scores, key=lambda x: x[1])
    print("****result********")
    for score in scores:
        print(score[0], ":", score[1])

    """
    精度微妙?

    画像のフォーマットによってもやっぱ代わりそう
    同じフォーマットの方が同じような評価になる

    ちょっと複雑な方がスコアまともに出やすいっぽい?
    """
