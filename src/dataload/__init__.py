

def setup(opt):
    if opt.dataset_name.lower() == 'funsd':
        from dataload.funsd import FUNSD as MyData
    elif opt.dataset_name.lower() == 'docvqa':
        from dataload.docvqa4lm import DocVQA as MyData
    elif opt.dataset_name.lower() == 'cord':
        from dataload.cord4layoutlm import CORD as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
