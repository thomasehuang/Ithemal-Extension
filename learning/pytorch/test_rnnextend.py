import torch
from models.graph_models import RnnParameters
from models.ithemal_extend import RNNExtend

if __name__ == '__main__':
    rnn_params = RnnParameters(
        embedding_size=256,
        hidden_size=256,
        num_classes=1,
        connect_tokens=False,           # NOT USED
        skip_connections=False,         # NOT USED
        hierarchy_type='MULTISCALE',    # NOT USED
        rnn_type='LSTM',                # NOT USED
        learn_init=True,                # NOT USED
    )
    model = RNNExtend(rnn_params)
    test_input = torch.rand(5, 1, 256)
    test_output = model(test_input)
    print(test_output.item())
