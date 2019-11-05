from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer

args = get_args_parser().parse_args(['-model_dir', '/tmp/uncased_L-12_H-768_A-12/',
                                     '-max_seq_len', 'NONE',
                                     '-pooling_strategy', 'NONE',
                                     '-device_map', '0', '1'])

server = BertServer(args)
server.start()
