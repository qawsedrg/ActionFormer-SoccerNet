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

data = SoccerNetTrainDataset("./SoccerNet", status="test")
test_loader = DataLoader(data, batch_size=1, collate_fn=lambda x: x)

model=torch.load("./model.pth")
model = nn.DataParallel(model, device_ids=["cuda:0"])

acc = []
for (i, video_list) in tqdm(enumerate(test_loader),total=len(test_loader)):
    video = video_list[0]
    model.eval()
    result = model(video_list)

    seg = result[0]["segments"].cpu().long()
    label = result[0]["labels"].cpu()
    bins = np.bincount(seg)
    bins_sorted = np.sort(bins)
    #
    acceptable_idx = np.argsort(bins)[bins_sorted >= max(bins_sorted[int(49 * len(bins) / 50)], 1)]

    pred_label = []
    pred_seg = []
    prediction = []
    for i in np.sort(acceptable_idx):
        label_bin = label[seg == i]
        label_pre = np.argmax(np.bincount(label_bin))
        pred_label.append(label_pre)
        pred_seg.append(i)
        prediction.append((i, label_pre))
    # print(prediction)

    tolerance = 10
    cen = video["segments"].numpy()
    l = video["labels"].numpy()
    l = np.tile(l[None, :], (len(pred_label), 1))
    pred_seg = np.array(pred_seg)
    pred_seg = np.tile(pred_seg[:, None], (1, len(cen)))
    pred_label = np.array(pred_label)
    pred_label = np.tile(pred_label[:, None], (1, len(cen)))
    left = cen - tolerance
    right = cen + tolerance
    mask = (pred_seg <= right) * (pred_seg >= left)
    # what if multiple points in one region of tolerance
    # true if one point within the tolerance region if correct, regardless of other points
    acc.append(np.average(np.clip(((pred_label == l) * mask).sum(0), 0, 1)))
    # print("acc : {:}".format(np.average(np.clip(((pred_label == l) * mask).sum(0), 0, 1))))
print(np.average(acc))