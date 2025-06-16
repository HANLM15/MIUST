import torch
from torch.utils.data import DataLoader
import os
from os.path import join
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
label_map = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
class IEMOCAPDataset( Dataset ):

    def __init__( self, audio_dir,text_dir, iemocap_csv,sessions ):
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self._sessions = sessions
        self.iemocap_csv = iemocap_csv

        self._sample_list = None

        self._audio_features = []
        self._text_features = []
        self._labels = []

    def construct_list( self ):
        csv_files = []
        for sess in self._sessions:
            text_label = pd.read_csv( join( self.iemocap_csv, "Session%d.csv" % sess ), 
                index_col = ["Utterance_ID"] )
            text_label["session"] = sess
            csv_files.append( text_label )
        
        self._sample_list = pd.concat( csv_files, axis = 0 )

    def construct_data( self ):
        for idx in range( len( self._sample_list ) ):
            Utterance_ID = self._sample_list.index[idx] 
            session = self._sample_list.iloc[idx]["session"]
            audio_feat = np.load( join( self.audio_dir, "Session%d/%s.npy" % ( session, Utterance_ID ) ) )
            audio_feat = audio_feat.astype( "float32" )
            text_feat = np.load( join( self.text_dir, "Session%d/%s.npy" % ( session, Utterance_ID ) ) )
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
    

def train_valid_test_iemocap_dataloader(audio_dir,text_dir,test_sess,batch_size,iemocap_csv,num_sess):
    sessions = list( range( 1, num_sess + 1 ) )
    # 测试和验证数据
    test_data = IEMOCAPDataset(audio_dir,text_dir,iemocap_csv,[test_sess])
    test_data.construct_list();test_data.construct_data()
    test_loader = DataLoader(test_data,batch_size=1,collate_fn=test_data.collator,shuffle=False,num_workers = 4)
    val_data = IEMOCAPDataset(audio_dir,text_dir,iemocap_csv,[test_sess])
    val_data.construct_list();val_data.construct_data()
    val_loader = DataLoader(val_data,batch_size=1,collate_fn=val_data.collator,shuffle=False,num_workers = 4)


    # 训练数据
    sessions.remove( test_sess )
    train_data = IEMOCAPDataset(audio_dir,text_dir,iemocap_csv,sessions)
    train_data.construct_list(); train_data.construct_data()
    train_loader = DataLoader(train_data,batch_size=batch_size,collate_fn=train_data.collator,shuffle=True,num_workers = 4)

    return  train_loader,val_loader,test_loader
 