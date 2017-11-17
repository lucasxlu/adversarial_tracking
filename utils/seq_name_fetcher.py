import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.config import *

if __name__ == '__main__':
    otb50_dir = configs['test_seq_base']

    option = {"sequence": {"type": "OTB50", "seq_names": None}}

    seq_name_list = []
    for _ in os.listdir(otb50_dir):
        seq_name_list.append(_)

    option['sequence']['seq_names'] = seq_name_list

    f = open('../tracking/options.json', mode='w')
    json.dump(option, f)
