import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="./SoccerNet")

#mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
# download Features reduced with PCA
#mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train","valid","test","challenge"])
#mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train","valid","test","challenge"])