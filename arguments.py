import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DEER Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--sdfa_lr', dest='sdfa_lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=2,
                        help='Number of graph convolution layers')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128,
                        help='Dimension of graph convolution layers')
    parser.add_argument('--aug', type=str, default='random4')
    parser.add_argument('--randperm', type=int, default=1)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--log_dir', default='log_dir', help='directory to save log')
    parser.add_argument('--log_file', type=str, default='results.txt', help='name of file for logging')
    parser.add_argument('--number_of_run', type=int, default=1)
    parser.add_argument('--start_upd_epoch', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--target_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--sdfa_eval_interval', type=int, default=1)
    parser.add_argument('--conv_type', type=str, default='GCN')
    parser.add_argument('--use_bn', type=int, default=1)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--global_pool', type=str, default='sum')
    parser.add_argument('--mmd_filter_ratio', type=float, default=0.7)
    parser.add_argument('--aug_strength', type=float, default=0.1)
    parser.add_argument('--num_aug', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--st_seed', type=int, default=0)
    parser.add_argument('--data_split', type=int, default=4)
    parser.add_argument('--source_index', type=int, default=0)
    parser.add_argument('--target_index', type=int, default=3)
    parser.add_argument('--cross_dataset', type=int, default=0)


    parser.add_argument('--use_teacher', type=int, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--cosistency_weight', type=float, default=0.2)
    parser.add_argument('--dual_teacher', type=int, default=1)
    parser.add_argument('--slow_teacher_speed', type=float, default=0.8)
    parser.add_argument('--use_motif_branch', type=int, default=1)
    parser.add_argument('--motif_weight', type=float, default=0.3)
    parser.add_argument('--use_mixup', type=int, default=1)
    parser.add_argument('--train_mixup', type=int, default=0)
    parser.add_argument('--train_mixup_weight', type=float, default=0.2)
    parser.add_argument('--mixup_weight_target', type=float, default=0.1)
    parser.add_argument('--mixup_weight_source', type=float, default=0.1)
    parser.add_argument('--teacher_dropout', type=float, default=0.3)
    parser.add_argument('--padto', type=int, default=0)
    parser.add_argument('--confident_rate', type=float, default=0.3)
    parser.add_argument('--motif_epochs', type=int, default=10)
    parser.add_argument('--motif_eval_interval', type=int, default=1)
    parser.add_argument('--update_psuedo', type=int, default=0)


    return parser.parse_args()

