import torch
from torch.utils.data import DataLoader
import os
from os.path import join
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

label_map = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
class MELDDataset( Dataset ):

    def __init__( self, audio_dir,text_dir, meld_csv):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.meld_csv = meld_csv

        self._sample_list = None

        self._audio_features = []
        self._text_features = []
        self._labels = []

    def construct_list( self ):
        self._sample_list = pd.read_csv( self.meld_csv, index_col = ["Filename"] )

    def construct_data( self ):
        for idx in range( len( self._sample_list ) ):
            Filename = self._sample_list.index[idx] 

            dir_name = os.path.basename(os.path.dirname(Filename))
            filename_withoutwav = os.path.splitext(os.path.basename(Filename))[0]
            audio_feat = np.load(join(self.audio_dir,dir_name,f"{filename_withoutwav}.npy"))
            audio_feat = audio_feat.astype( "float32" )
            text_feat = np.load(join(self.text_dir,dir_name,f"{filename_withoutwav}.npy"))
            text_feat = text_feat.astype( "float32" )
            label = label_map[self._sample_list.iloc[idx]["Emotion"]]

            self._audio_features.append( audio_feat )
            self._text_features.append( text_feat )
            self._labels.append( label )
            
        self._sample_list = None

    def __getitem__( self, idx ):
        return self._text_features[idx], self._audio_features[idx],self._labels[idx]

    def __len__( self ):
        return len( self._labels )
    
    def collator(self, samples):
            if len(samples) == 0:
                return {}

            text_feats = [s[0] for s in samples]
            audio_feats = [s[1] for s in samples]
            labels = torch.tensor([s[2] for s in samples])

            text_sizes = [s.shape[0] for s in text_feats]
            audio_sizes = [s.shape[0] for s in audio_feats]
            
            max_text_size = max(text_sizes)
            max_audio_size = max(audio_sizes)
            
            text_feat_dim = text_feats[0].shape[1]
            audio_feat_dim = audio_feats[0].shape[1]

            collated_text_feats = torch.zeros((len(text_feats), max_text_size, text_feat_dim), dtype=torch.float32)
            collated_audio_feats = torch.zeros((len(audio_feats), max_audio_size, audio_feat_dim), dtype=torch.float32)

            text_padding_mask = torch.BoolTensor(len(text_feats), max_text_size).fill_(False)
            audio_padding_mask = torch.BoolTensor(len(audio_feats), max_audio_size).fill_(False)

            for i, (text_feat, audio_feat, text_size, audio_size) in enumerate(zip(text_feats, audio_feats, text_sizes, audio_sizes)):
                collated_text_feats[i, :text_size] = torch.from_numpy(text_feat)
                text_padding_mask[i, text_size:] = True
                collated_audio_feats[i, :audio_size] = torch.from_numpy(audio_feat)
                audio_padding_mask[i, audio_size:] = True

            return {
                "id": torch.LongTensor([i for i in range(len(samples))]),
                "net_input": {
                    "text_feats": collated_text_feats,
                    "audio_feats": collated_audio_feats,
                    "text_padding_mask": text_padding_mask,
                    "audio_padding_mask": audio_padding_mask
                },
                "labels": labels
            }
    

def train_valid_test_meld_dataloader(audio_dir,text_dir,batch_size,meld_csv):
    # 测试和验证数据
    test_data = MELDDataset(audio_dir,text_dir,meld_csv["test_csv"])
    test_data.construct_list();test_data.construct_data()
    test_loader = DataLoader(test_data,batch_size=1,collate_fn=test_data.collator,shuffle=False,num_workers = 4)
    dev_data = MELDDataset(audio_dir,text_dir,meld_csv["dev_csv"])
    dev_data.construct_list();dev_data.construct_data()
    dev_loader = DataLoader(dev_data,batch_size=1,collate_fn=dev_data.collator,shuffle=False,num_workers = 4)

    # 训练数据
    train_data = MELDDataset(audio_dir,text_dir,meld_csv["train_csv"])
    train_data.construct_list(); train_data.construct_data()
    train_loader = DataLoader(train_data,batch_size=batch_size,collate_fn=train_data.collator,shuffle=True,num_workers = 4)

    return  train_loader,dev_loader,test_loader
 