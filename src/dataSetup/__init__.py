

def setup(opt):
    if opt.dataset_name.lower() == 'funsd4g':
        from dataSetup.funsd4g import FUNSD as MyData
    elif opt.dataset_name.lower() == 'funsd4lm':
        from dataSetup.funsd4lm2 import FUNSD as MyData
    elif opt.dataset_name.lower() == 'docvqa4lm':
        from dataSetup.docvqa4lm import DocVQA as MyData
    elif opt.dataset_name.lower() == 'cord4lm':
        from dataSetup.cord4lm2 import CORD as MyData
    elif opt.dataset_name.lower() == 'cord4g':
        from dataSetup.cord4g import CORD4G as MyData
    else:
        raise Exception('dataset not supported:{}'.format(opt.dataset_name))

    mydataset = MyData(opt=opt)
    return mydataset
