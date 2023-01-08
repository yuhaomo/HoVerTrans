import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--model_name', type=str, default='hovertrans')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--writer_comment', type=str, default='GDPH&SYSUCC')
    parser.add_argument('-s', '--save_model_in_epoch', action='store_true', default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--patch_size', type=list, default=[2, 2, 2, 2])
    parser.add_argument('--hover_size', type=list, default=[2, 2, 2, 2])
    parser.add_argument('--dim', type=list, default=[4, 8, 16, 32])
    parser.add_argument('--depth', type=list, default=[2, 4, 4, 2])
    parser.add_argument('--num_heads', type=list, default=[2, 4, 8, 16])
    parser.add_argument('--num_inner_head', type=list, default=[2, 4, 8, 16])

    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)


    config = parser.parse_args()
    return config