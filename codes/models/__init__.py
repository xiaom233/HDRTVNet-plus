import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'base':
        from .Generation_base_model import GenerationModel as M
    elif model == 'condition':
        from .Generation_condition_model import GenerationModel as M
    elif model == 'hallucination':
        from .Generation_hallucination_model import GenerationModel as M
    elif model == 'hallucination_GAN':
        from .Generation_hallucination_GAN_model import GenerationModel as M
    elif model == 'hallucination_mask_GAN':
        from .Generation_hallucination_mask_GAN_model import GenerationModel as M
    elif model == 'hallucination_maskedDiscriminator_GAN':
        from .Generation_hallucination_maskedDiscriminator_GAN_model import GenerationModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
