import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="two steam model for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    # ========================= Data Configs ==========================
    parser.add_argument('--unlabeled_annotation', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
    parser.add_argument('--unlabeled_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/unlabeled/')
    parser.add_argument('--unlabeled_zip_feats', type=str, default='/home/tione/notebook/data/zip_feats/unlabeled.zip')

    parser.add_argument('--train_annotation', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')
    parser.add_argument('--train_zip_feats', type=str, default='/home/tione/notebook/data/zip_feats/labeled.zip')

    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')



    parser.add_argument('--log_path', type=str, default=f'../log')
    parser.add_argument('--model_path', type=str, default=f'../models')
    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='../opensource_models/swin_base_patch4_window7_224_22k.pth')
    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='../opensource_models/chinese-macbert-base')

    # parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='../cache')
    parser.add_argument('--bert_seq_lenght', type=int, default=256)
    #
    parser.add_argument('--n_splits', default=0, type=float, help='split 10')
    parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=2, type=int, help="num_workers for dataloaders")

    # DDP
    # DDP
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    # ========================== tricks =============================
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_adv', type=int, default=0, help="0-not use, 1-fpm, 2-pdg")
    # ========================== 重新加载模型 =============================
    parser.add_argument('--resume', type=str, default=f'')

    # inference
    parser.add_argument('--ckpt_file', type=str, default=f'../models/finetune/49.0/model_fold_0_epoch_3_step_6250.bin')
    args = parser.parse_args()
    return args
