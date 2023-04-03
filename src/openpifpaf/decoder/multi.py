import logging
from typing import List

from .decoder import Decoder
from ..network.nets import multi_apply

LOG = logging.getLogger(__name__)


class Multi(Decoder):
    def __init__(self, decoders):
        super().__init__()

        self.decoders = decoders

    def __call__(self, all_fields):
        out = []
        print('all fields length: {}'.format(len(all_fields)))
        print('decoders type: {}, decoders[0] type: {}'.format(type(self.decoders),type(self.decoders[0])))
        for task_i, decoder in enumerate(self.decoders):
            if decoder is None:
                out.append(None)
                continue
            LOG.debug('task %d', task_i)
            if type(decoder)==tuple:
                assert len(decoder)==len(all_fields)
                for i in range(len(decoder)):
                    out+=decoder[i](all_fields[i])
            else:
                out += decoder(all_fields)

        return out

    def reset(self):
        # TODO: remove?
        for dec in self.decoders:
            if not hasattr(dec, 'reset'):
                continue
            dec.reset()

    @classmethod
    def factory(cls, head_metas) -> List['Generator']:
        raise NotImplementedError
