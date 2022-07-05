import os
import json
from tqdm import tqdm
import numpy as np

from modeling.meta_archs import *

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from SoccerNet.Downloader import getListGames

from torch.utils.data import Dataset,DataLoader

class SoccerNetTrainDataset(Dataset):
    def __init__(self,
                 feat_file_address,
                 features="ResNET_TF2_PCA512.npy",
                 status="train",
                 framerate=2
                 ):
        assert os.path.exists(feat_file_address)
        self.path = feat_file_address
        self.features = features
        self.listGames = getListGames(status)
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = len(EVENT_DICTIONARY_V2)
        self.labels = "Labels-v2.json"
        self.game_id = list()
        self.game_segs = list()
        self.game_feats = list()
        self.game_labels = list()
        for game in tqdm(self.listGames):
            id1 = os.path.join(self.path, game, "1_" + self.features)
            feat_half1 = torch.from_numpy(np.load(id1)).permute((1, 0))
            id2 = os.path.join(self.path, game, "2_" + self.features)
            feat_half2 = torch.from_numpy(np.load(id2)).permute((1, 0))
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            seg_half1 = []
            seg_half2 = []
            label_half1 = []
            label_half2 = []
            for annotation in labels["annotations"]:
                if annotation["visibility"] != "visible":
                    continue
                gameTime = annotation["gameTime"].split("-")
                event = annotation["label"]
                half = int(gameTime[0])
                time = gameTime[1].split(":")
                frame = int(time[0]) * 60 * 2 + int(time[1]) * 2
                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]
                if half == 1:
                    label_half1.append(label)
                    seg_half1.append(frame)
                if half == 2:
                    label_half2.append(label)
                    seg_half2.append(frame)

            self.game_id.append(id1)
            self.game_id.append(id2)
            self.game_segs.append(torch.Tensor(seg_half1))
            self.game_segs.append(torch.Tensor(seg_half2))
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(torch.Tensor(label_half1).long())
            self.game_labels.append(torch.Tensor(label_half2).long())

    def __getitem__(self, index):
        return {"video_id": self.game_id[index],
                "feats": self.game_feats[index],
                "segments": self.game_segs[index],
                "labels": self.game_labels[index]}

    def __len__(self):
        return len(self.game_feats)

if True:
    data = SoccerNetTrainDataset("./SoccerNet", status="test")
    test_loader = DataLoader(data, batch_size=2, collate_fn=lambda x: x)

    model = torch.load("./model.pth")
    model = nn.DataParallel(model, device_ids=["cuda:0"])
    for (i, video_list) in tqdm(enumerate(test_loader),total=len(test_loader)):
        predict_dict = dict()
        predictions = []
        for video in video_list:
            half = video["video_id"].split("/")[-1][0]
            predict_dict["UrlLocal"] = "/".join(video["video_id"].split("/")[2:5])
            model.eval()
            result = model([video])

            seg = result[0]["segments"].cpu().long()
            label = result[0]["labels"].cpu()
            scores = result[0]["scores"].cpu()
            bins = np.bincount(seg)
            bins_sorted = np.sort(bins)
            #
            acceptable_idx = np.argsort(bins)[bins_sorted >= max(bins_sorted[int(19 * len(bins) / 20)], 1)]
            acceptable_idx=np.sort(acceptable_idx)

            max_interval=2
            counter=0
            while counter<len(acceptable_idx):
                idx=acceptable_idx[counter]
                prediction_dict=dict()
                label_bin=np.array([],dtype=int)
                pred_seg = []
                scores_bin=np.array([])
                while np.sum(acceptable_idx[counter:]<=idx+max_interval)>=1:
                    label_bin = np.concatenate((label_bin,label[seg == acceptable_idx[counter]]))
                    # how to define score in this case?
                    pred_seg.extend([acceptable_idx[counter]]*len(label[seg == acceptable_idx[counter]]))
                    scores_bin=np.concatenate((scores_bin,scores[seg == acceptable_idx[counter]].numpy()))
                    idx = acceptable_idx[counter]
                    counter+=1
                label_pre = np.argmax(np.bincount(label_bin))
                mask=label_pre==label_bin
                scores_pred=np.mean(scores_bin[mask])
                frame=np.mean(pred_seg)
                prediction_dict["gameTime"] = "{} - {}:{}".format(half,int((frame // 2)) // 60, int((frame // 2)) % 60 if (frame // 2) % 60>=10 else "0{}".format(int((i // 2) % 60)))
                prediction_dict["label"] = dict(zip(EVENT_DICTIONARY_V2.values(), EVENT_DICTIONARY_V2.keys()))[label_pre]
                prediction_dict["position"] = str(int(frame/2*1000))
                prediction_dict["half"] = half
                prediction_dict["confidence"] = str(scores_pred)
                predictions.append(prediction_dict)
        predict_dict["predictions"]=predictions
        if not os.path.exists(predict_dict["UrlLocal"]):
            os.makedirs(predict_dict["UrlLocal"])
        with open("{}/results_spotting.json".format(predict_dict["UrlLocal"]), "w") as f:
            json.dump(predict_dict, f, indent=2)

from SoccerNet.Evaluation.ActionSpotting import evaluate
results = evaluate(SoccerNet_path="./SoccerNet", Predictions_path="./",
                   split="test", version=2, prediction_file="results_spotting.json", metric="tight")
print("tight Average mAP: ", results["a_mAP"])
print("tight Average mAP per class: ", results["a_mAP_per_class"])
print("tight Average mAP visible: ", results["a_mAP_visible"])
print("tight Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("tight Average mAP unshown: ", results["a_mAP_unshown"])
print("tight Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])