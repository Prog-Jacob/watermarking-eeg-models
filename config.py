import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Specify config args for watermarking EEG-based neural networks experiments and evaluations.')
    
    parser.add_argument('--evaluate', nargs='+', help='Specify any number of dimensions to evaluate from correct_watermark, wrong_watermark, new_watermark, or eeg!', default = ['correct_watermark', 'wrong_watermark', 'new_watermark', 'eeg'] , required=False)
    parser.add_argument('--experiment', help='Specify any number of experiments to do from nowatermark, newwatermark:{fromscratch,pretrain}, fromscratch, pretrain, pruning:{ascending,descending,random}, quantization, transfer_learning:{all,added,dense}, or fine_tuning:{ftll,ftal,rtll,rtal}!', required=True)
    parser.add_argument('--architecture', help='Choose CCNN, EEGNet, or TSCeption!', default = "CCNN", required=True)
    parser.add_argument('--skip_training', dest='skip_training', action='store_true')

    parser.add_argument('--root_path', help='Provide the path to the processed python data directory!', default = './data/data_processed_python', required=False)
    parser.add_argument('--batch', type = int, help='Number of samples per batch!', required=False)
    parser.add_argument('--epochs', type = int, help='Number of epochs!', required=False)
    parser.add_argument('--lrate', type = float, help='Learning rate!', required=False)
        
    args = vars(parser.parse_args())
    return args