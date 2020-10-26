from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, sys, time


# APIキーの情報
key = "0f1879729eaffa3d061b8fdeb1b51e5a"   # public key
seacret = "13994d7bda49bfea"               # seacret key
wait_time = 1                              # flickrにアクセスする間隔

# 保存フォルダの指定
animalname = sys.argv[1]  # コマンドライン引数　monkey, boar, crowのいずれかを入れる
savedir = "./" + animalname

flickr = FlickrAPI(key, seacret, format='parsed-json')  # json形式で返す
result = flickr.photos.search(
	text = animalname,
	per_page = 400,             # 外れ値覗いて300程欲しい
	media = 'photos',           # 検索するデータの種類
	sort = 'relevance',         # 関連順
	safe_search = 1,            # 有害コンテンツは表示しない
	extras = 'url_q, licence'   # 取得したいオプション値　url_q:画像のアドレス　licence:ライセンス
)

# 結果の表示
photos = result['photos']  # このkeyは？
# pprint(photos)

# データの保存
for i, photo in enumerate(photos['photo']):
	url_q = photo['url_q']
	filepath = savedir + '/' + photo['id'] + '.jpg'
	# ファイルが重複確認
	if os.path.exists(filepath):
		continue
	urlretrieve(url_q, filepath)  # retrieve 取り出す
	time.sleep(wait_time)         # アクセス制御

