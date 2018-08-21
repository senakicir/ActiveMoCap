from configobj import ConfigObj
import os


def config_reader():
    """
    Returns the data contained in the configuration file in the form of a dictionary.
    The values are converted when needed.
    """
    config = ConfigObj(os.path.join(os.path.dirname(__file__), 'config'))

    param = config['param']
    model_id = param['modelID']
    model = config['models'][model_id]
    model['module'] = '.' + model['module'] if '.' not in model['module'][0] else model['module']
    model['nbHM'] = int(model['nbHM'])
    model['nbJoints'] = int(model['nbJoints'])
    model['indexHM'] = list(map(int, model['indexHM']))
    model['boxsize'] = int(model['boxsize'])
    model['marginBox'] = float(model['marginBox'])
    model['square'] = True if model['square'] in 'True' else False
    model['pad'] = True if model['pad'] in 'True' else False
    model['padValue'] = int(model['padValue'])

    return param, model
