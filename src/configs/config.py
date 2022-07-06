import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="two steam model for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')

    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')

    parser.add_argument('--train_feature_output_path', type=str, default='/home/tione/notebook/data/zip_feats/labeled/')



    parser.add_argument('--log_path', type=str, default=f'../log')
    parser.add_argument('--model_path', type=str, default=f'../models')
    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='../opensource_models/swin_tiny_patch4_window7_224.pth')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='../opensource_models/chinese-macbert-base')

    # parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_cache', type=str, default='../cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    #
    parser.add_argument('--n_splits', default=5, type=float, help='split 10')
    parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=2, type=int, help="num_workers for dataloaders")

    # DDP
    parser.add_argument(
        "--nodes", default=1, type=int, help="number of nodes for distributed training"
    )
    parser.add_argument(
        "--ngpus_per_node",
        default=2,
        type=int,
        help="number of GPUs per node for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:12306",
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--node_rank", default=0, type=int, help="node rank for distributed training"
    )

    # ========================== tricks =============================
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_adv', type=int, default=0, help="0-not use, 1-fpm, 2-pdg")
    # ========================== 重新加载模型 =============================
    parser.add_argument('--resume', type=str, default=f'')

    # inference
    parser.add_argument('--ckpt_file', type=str, default=f'../models/finetune/1.0/model_fold_0_epoch_1_step_100.bin')
    args = parser.parse_args()
    return args
