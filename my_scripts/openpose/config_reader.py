from configobj import ConfigObj
import os


def config_reader():
    config = ConfigObj(os.path.join(os.path.dirname(__file__), 'config'))

    param = config['param']
    model_id = param['modelID']
    param['scale_search'] = list(map(float, param['scale_search']))
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])

    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])

    return param, model
