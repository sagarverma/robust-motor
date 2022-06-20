import argparse as ag


def get_args():
    parser = ag.ArgumentParser(description='Training or inference')

    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        required=False,
                        help="GPU ID on which to run")
                        
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset Name')

    parser.add_argument('--model',
                        type=str,
                        required=True)
    
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        required=False,
                        help='Number of epochs to train.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        required=False,
                        help='Training or test batch size.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        required=False,
                        help='Number of cpu cores to use')

    args = parser.parse_args()
    return args