import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data

from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.cityscapes import CityscapesDataSet
from advent.dataset.mapillary import MapillaryDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.esl_UDA import extract_pseudo_labels

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'

    print('Using config:')
    pprint.pprint(cfg)
    # load model
    model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                           multi_level=cfg.ESL.MULTI_LEVEL)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    if cfg.TARGET == 'Mapillary':
        test_dataset = MapillaryDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                          set=cfg.ESL.SET_TARGET,
                                          crop_size=cfg.ESL.INPUT_SIZE_TARGET,
                                          mean=cfg.ESL.IMG_MEAN,
                                          labels_size=cfg.ESL.OUTPUT_SIZE_TARGET)
    else:
        test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                         list_path=cfg.DATA_LIST_TARGET,
                                         set=cfg.ESL.SET_TARGET,
                                         info_path=cfg.ESL.INFO_TARGET,
                                         crop_size=cfg.ESL.INPUT_SIZE_TARGET,
                                         mean=cfg.ESL.IMG_MEAN,
                                         labels_size=cfg.ESL.OUTPUT_SIZE_TARGET,
                                         num_classes=cfg.NUM_CLASSES)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.ESL.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    # eval
    extract_pseudo_labels(model, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)
