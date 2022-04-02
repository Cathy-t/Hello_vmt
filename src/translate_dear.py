# -*- coding: utf-8 -*-

# @Time : 2021/5/23 18:50
# @Author : Cathy
# @FileName: translate_dear.py
# @Software: PyCharm

import sys
import os
import argparse
import time
import datetime
import logging
import numpy as np
import json

import torch
import torch.nn as nn

import sys
sys.path.append(r'E:\graduation\code\demo\VMT\src')

from dear import make_model
from utils import padding_idx, sos_idx, eos_idx, unk_idx, beam_search_decode
from utils import read_vocab, write_vocab, build_vocab, Tokenizer
import random


class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def setup(args):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    #build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')  # encoding='latin1' to handle the inconsistency between python 2 and 3
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)  # 前feats.shape[0]行，为真正的特征，且顺序没有变
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))   # inds对应的类别需要被记录下来
        feats = feats[inds]
    assert feats.shape[0] == max_length
    img = torch.from_numpy(np.float32(feats))  # torch.Size([32, 1024])

    # 构建img_mask
    img_mask = (torch.sum(img != 1, dim=1) != 0).unsqueeze(-2)  # torch.Size([1, 32])
    img = img * img_mask.squeeze().unsqueeze(-1).expand_as(img).float()  # torch.Size([32, 1024])

    return img, img_mask


def translate(video_name, src):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='E:\graduation\code\demo\VMT\src\configs_dear.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.load(fin))

    tgt_sentence = main(args, video_name, src)
    return tgt_sentence


def main(args, video_name, src):

    checkpoint_path = args.CHK_DIR
    cp_file = os.path.join(checkpoint_path, 'best_model_dear.pth.tar')

    if not os.path.exists(checkpoint_path):
        sys.exit('No checkpoint_path found {}'.format(checkpoint_path))

    # set up vocab txt
    setup(args)
    print(args.__dict__)

    # indicate src and tgt language
    en_input, zh_input = 'en', 'zh'

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}
    vocab_en = read_vocab(maps[en_input])
    tok_en = Tokenizer(language=en_input, vocab=vocab_en, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_zh = read_vocab(maps[zh_input])
    tok_zh = Tokenizer(language=zh_input, vocab=vocab_zh, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size en/zh:{}/{}'.format(len(vocab_en), len(vocab_zh)))

    ## init model
    model = make_model(len(vocab_en), len(vocab_zh), N=args.nb_blocks, d_model=args.d_model, d_ff=args.d_model * 4,
                       h=args.att_h, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    """eval"""
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))
    model.load_state_dict(torch.load(cp_file)['state_dict'])

    model.eval()  # eval mode (no dropout or batchnorm)

    with torch.no_grad():
        print(src)
        encap, ensrccap_mask, _, _ = tok_en.encode_sentence(src)
        video, video_mask = load_video_features(os.path.join(args.DATA_DIR, video_name + '.npy'), args.MAX_VID_LENGTH)

        encap, ensrccap_mask, video, video_mask = encap.cuda(), ensrccap_mask.cuda(), video.cuda(), video_mask.cuda()

        nbest = 1

        # en2zh
        pred_out, _ = beam_search_decode(model, encap, ensrccap_mask, video, video_mask, args.maxlen,
                                         start_symbol=sos_idx,
                                         unk_symbol=unk_idx, end_symbol=eos_idx,
                                         pad_symbol=padding_idx)

        for n in range(min(nbest, len(pred_out))):
            pred = pred_out[n]
            temp_preds = []
            for w in pred[0]:
                if w == eos_idx:
                    break
                temp_preds.append(w)
            if n == 0:
                preds = tok_zh.decode_sentence(temp_preds)
                tgt = preds
    print(tgt)
    return tgt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs_dear.yaml')
    parser.add_argument('--video_name', type=str, default='_0nX-El-ySo_83_93')
    parser.add_argument('--src', type=str, default='a man is cutting a piece of paper')

    args = parser.parse_args()
    video_name = args.video_name
    src = args.src
    with open(args.config, 'r') as fin:
        import yaml

        args = Arguments(yaml.load(fin))
    main(args, video_name, src)