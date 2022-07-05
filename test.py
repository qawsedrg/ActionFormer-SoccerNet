import numpy as np
import torch
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

model = torch.load("./model_2048.pth")
model.cuda()
model.eval()
name = "./example_soccer_woPCA.npy"
video_list = [{"video_id": name,
               "feats": torch.from_numpy(np.load(name)).permute((1, 0)),
               "segments": torch.Tensor(),
               "labels": torch.Tensor().long()}]
result = model(video_list)

seg = result[0]["segments"].cpu().long()
label = result[0]["labels"].cpu()
bins = np.bincount(seg)
bins_sorted = np.sort(bins)
#
acceptable_idx = np.argsort(bins)[bins_sorted >= max(bins_sorted[int(29 * len(bins) / 30)], 1)]

pred_label = []
pred_seg = []
prediction = []
for i in np.sort(acceptable_idx):
    label_bin = label[seg == i]
    label_pre = np.argmax(np.bincount(label_bin))
    pred_label.append(label_pre)
    pred_seg.append(i)
    prediction.append(
        ((i // 2) // 60, (i // 2) % 60, dict(zip(EVENT_DICTIONARY_V2.values(), EVENT_DICTIONARY_V2.keys()))[label_pre]))
print(prediction)
