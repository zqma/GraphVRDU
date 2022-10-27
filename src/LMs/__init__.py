from transformer_models.CNN import TextCNN
from transformer_models.MLP import MLP
from transformer_models.DistilBert import DistilBERTClassifier
from transformer_models.Roberta import RobertaClassifier


def setup(opt):
    print('network:' + opt.network_type)
    if opt.network_type == 'cnn':
        model = TextCNN(opt)
    elif opt.network_type == 'mlp':
        model = MLP(opt)
    elif opt.network_type == 'distilbert':
        model = DistilBERTClassifier(opt)
    elif opt.network_type == 'roberta':
        model = RobertaClassifier(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model