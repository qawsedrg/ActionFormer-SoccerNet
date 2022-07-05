import os
import json
from tqdm import tqdm

from modeling.meta_archs import *
from modeling.utils import make_optimizer, make_scheduler

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from SoccerNet.Downloader import getListGames

from torch.utils.data import Dataset,DataLoader

cfg_model = {'fpn_type': 'identity', 'max_buffer_len_factor': 6.0, 'n_mha_win_size': 19,
             'backbone_type': 'convTransformer',
             'backbone_arch': (2, 2, 5), 'scale_factor': 2,
             'regression_range': [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)], 'n_head': 4,
             'embd_kernel_size': 3,
             'embd_dim': 512, 'embd_with_ln': True, 'fpn_dim': 512, 'fpn_with_ln': True, 'head_dim': 512,
             'head_kernel_size': 3,
             'head_num_layers': 3, 'head_with_ln': True, 'use_abs_pe': False, 'use_rel_pe': False, 'input_dim': 512,
             'num_classes': 17, 'max_seq_len': 6912,
             'train_cfg': {'init_loss_norm': 100, 'clip_grad_l2norm': 1.0, 'cls_prior_prob': 0.01,
                           'center_sample': 'radius',
                           'center_sample_radius': 1.5, 'loss_weight': 1.0, 'head_empty_cls': [], 'dropout': 0.0,
                           'droppath': 0.1,
                           'label_smoothing': 0.0},
             'test_cfg': {'voting_thresh': 0.7, 'pre_nms_topk': 2000, 'max_seg_num': 200, 'min_score': 0.001,
                          'multiclass_nms': True, 'pre_nms_thresh': 0.001, 'iou_threshold': 0.1, 'nms_method': 'soft',
                          'nms_sigma': 0.5, 'duration_thresh': 0.05, 'ext_score_file': None}}

cfg_opt = {'learning_rate': 0.0001, 'epochs': 30, 'weight_decay': 0.05, 'type': 'AdamW', 'momentum': 0.9,
           'warmup': True, 'warmup_epochs': 5, 'schedule_type': 'cosine', 'schedule_steps': [], 'schedule_gamma': 0.1}

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

data = SoccerNetTrainDataset("./SoccerNet", status="train")
train_loader = DataLoader(data, batch_size=2, collate_fn=lambda x: x)

model = PtTransformer(**cfg_model)
model = nn.DataParallel(model, device_ids=["cuda:0"])
optimizer = make_optimizer(model, cfg_opt)
scheduler = make_scheduler(optimizer, cfg_opt, num_iters_per_epoch=len(train_loader))
clip_grad_l2norm = 1

for epoch in range(30):
    for (i, video_list) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = model(video_list)
        losses['final_loss'].backward()
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad_l2norm)
        optimizer.step()
        scheduler.step()

        block3 = 'epoch {:} {:}/{:} Loss {:.2f}'.format(epoch, i, len(train_loader),
                                                        losses['final_loss'].item())
        print('\t'.join([block3]))