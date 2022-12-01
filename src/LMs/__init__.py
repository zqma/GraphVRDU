
# from LMs.Roberta import RobertaClassifier
from LMs.LayoutLM import LayoutLMTokenclassifier

def setup(opt):
    print('network:' + opt.network_type)
    # if opt.network_type == 'roberta':
    #     model = RobertaClassifier(opt)
    if opt.network_type == 'layoutlm':
        model = LayoutLMTokenclassifier(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model