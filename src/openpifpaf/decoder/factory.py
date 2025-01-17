from collections import defaultdict
import logging
import copy
from typing import Optional

from .cifcaf import CifCaf, CifCafDense
from .cifdet import CifDet
from .decoder import Decoder
from .multi import Multi
from .pose_similarity import PoseSimilarity
from .track_base import TrackBase
from .tracking_pose import TrackingPose
from . import utils
from ..profiler import Profiler  # , TorchProfiler
from ..network.nets import multi_apply

LOG = logging.getLogger(__name__)

DECODERS = {CifDet, CifCaf, CifCafDense, PoseSimilarity, TrackingPose}


def cli(parser, *, workers=None):
    group = parser.add_argument_group('decoder configuration')

    available_decoders = [dec.__name__.lower() for dec in DECODERS]
    group.add_argument('--decoder', default=None, nargs='+',
                       help='Decoders to be considered: {}.'.format(available_decoders))

    assert utils.CifSeeds.get_threshold() == utils.CifDetSeeds.get_threshold()
    group.add_argument('--seed-threshold', default=utils.CifSeeds.get_threshold(), type=float,
                       help='minimum threshold for seeds')
    assert CifDet.instance_threshold == utils.nms.Keypoints.get_instance_threshold()
    group.add_argument('--instance-threshold', type=float, default=None,
                       help=('filter instances by score (default is 0.0 with '
                             '--force-complete-pose and {} otherwise)'
                             ''.format(utils.nms.Keypoints.get_instance_threshold())))
    group.add_argument('--decoder-workers', default=workers, type=int,
                       help='number of workers for pose decoding')

    group.add_argument('--profile-decoder', nargs='?', const='profile_decoder.prof', default=None,
                       help='specify out .prof file or nothing for default file name')

    group = parser.add_argument_group('CifCaf decoders')
    group.add_argument('--cif-th', default=utils.CifHr.get_threshold(), type=float,
                       help='cif threshold')
    group.add_argument('--caf-th', default=utils.CafScored.get_default_score_th(), type=float,
                       help='caf threshold')

    #add head_stride
    group.add_argument('--base_stride', default=None, nargs='+',type=int,
                           help='Config dict for the strides of headnet')

    TrackBase.cli(parser)
    for dec in DECODERS:
        dec.cli(parser)


def configure(args):
    if args.instance_threshold is None:
        if args.force_complete_pose:
            args.instance_threshold = 0.0
        else:
            args.instance_threshold = utils.nms.Keypoints.get_instance_threshold()

    # configure Factory
    Factory.decoder_request_from_args(args.decoder)
    Factory.profile = args.profile_decoder

    # configure CifHr
    utils.CifHr.set_threshold(args.cif_th)

    # configure CifSeeds
    utils.CifSeeds.set_threshold(args.seed_threshold)
    utils.CifDetSeeds.set_threshold(args.seed_threshold)

    # configure CafScored
    utils.CafScored.set_default_score_th(args.caf_th)

    # configure generators
    Decoder.default_worker_pool = args.decoder_workers

    # configure instance threshold
    utils.nms.Keypoints.set_instance_threshold(args.instance_threshold)
    CifDet.instance_threshold = args.instance_threshold


    #add head_stride
    Factory.base_stride = args.base_stride




    TrackBase.configure(args)
    for dec in DECODERS:
        dec.configure(args)


class Factory:
    decoder_request: Optional[dict] = None
    profile = False
    base_stride = None

    @classmethod
    def decoder_request_from_args(cls, list_str):
        if list_str is None:
            cls.decoder_request = None
            return

        cls.decoder_request = defaultdict(list)
        for dec_str in list_str:
            # pylint: disable=unsupported-assignment-operation,unsupported-membership-test,unsubscriptable-object
            if ':' not in dec_str:
                if dec_str not in cls.decoder_request:
                    cls.decoder_request[dec_str] = []
                continue

            dec_str, _, index = dec_str.partition(':')
            index = int(index)
            cls.decoder_request[dec_str].append(index)   #We could have several instances for each kind of decoder(eg. 3cif,3caf,...)

        LOG.debug('setup decoder request: %s', cls.decoder_request)

    @classmethod
    def decoders(cls, head_metas):
        def per_class(request, dec_class):
            class_name = dec_class.__name__.lower()
            if request is not None \
               and class_name not in request:
                return []
            decoders = sorted(dec_class.factory(head_metas), key=lambda d: d.priority, reverse=True)
            # print(len(decoders))
            for dec_i, dec in enumerate(decoders):
                dec.request_index = dec_i
            if request is not None:
                indices = set(request[class_name])
                # print(request)
                # print(indices)
                decoders = (d for i, d in enumerate(decoders) if i in indices)
            return decoders

        decoders = [d for dec_class in DECODERS for d in per_class(cls.decoder_request, dec_class)]
        decoders = list(sorted(decoders, key=lambda d: d.priority, reverse=True))
        LOG.debug('created %d decoders', len(decoders))

        #Since we will clarify decoder_request in the cli function(also in val config files), we will not enter the following if-else statement
        if not decoders:
            LOG.warning('no decoders found for heads %s', [meta.name for meta in head_metas])
        elif len(decoders) == 1:
            pass
        elif cls.decoder_request is None:
            LOG.info(
                'No specific decoder requested. Using the first one from:\n'
                '%s\n'
                'Use any of the above arguments to select one or multiple '
                'decoders and to suppress this message.',
                '\n'.join(
                    f'  --decoder={dec.__class__.__name__.lower()}:{dec.request_index}'
                    for dec in decoders
                )
            )
            decoders = [decoders[0]]

        return decoders

    @classmethod
    def __call__(cls, head_metas):
        """Instantiate decoders."""
        LOG.debug('head names = %s', [meta.name for meta in head_metas])
        # print('original head_meta length:{}'.format(len(head_metas)))
        # for meta in head_metas:
        #     print(meta)
        
        # print(type(cls.base_stride))
        if type(cls.base_stride) == list:
            new_head_meta = []
            for hs in cls.base_stride:
                single_stage_hm = []
                for hm in head_metas:
                    new_hm = copy.deepcopy(hm)
                    new_hm.base_stride = hs
                    # print(new_hm)
                    single_stage_hm.append(new_hm)
                new_head_meta.append(single_stage_hm)
            new_head_meta = tuple(new_head_meta)
            # decoders = cls.decoders(new_head_meta)
            decoders = multi_apply(cls.decoders,new_head_meta) 
            ##new_head_meta would be tuple([cif1,caf1,cifdet1],[cif2,caf2,cifdet2]), in multi_apply, this tuple will first be unpacked to k=len(stride) list 
            ##and the cls.decoders will be applied to each list in parallel. Each will return a result decoder list [CifCaf, Cifdet] with the corresponding stride
            ##Then the zip(*map_results) in multi_apply will zip the k decoder outputs to a form of (CifCaf1,CifCaf2,...),(CifDet1,CifDet2,...) 
            ##and map(list, zip(*map_results)) will map them into two lists [CifCaf1,CifCaf2,...],[CifDet1,CifDet2,...]
            ##Finally, tuple(map(list, zip(*map_results))) will put these two lists in a tuple
        else:
            decoders = cls.decoders(head_metas)
  

        # print(type(decoders),type(decoders[0]))
        # print(len(decoders),len(decoders[0]))
        # for dec in decoders[0]:
        #     print(dec)
        #     print(dec.cif_metas)
        #     print(dec.caf_metas)
        # for dec in decoders[1]:
        #     print(dec)
        #     print(dec.metas)

        if cls.profile:
            decode = decoders[0]
            decode.__class__.__call__ = Profiler(
                decode.__call__, out_name=cls.profile)
            # decode.__class__.__call__ = TorchProfiler(
            #     decode.__call__, out_name=cls.profile)

        return Multi(decoders)

        # TODO implement!
        # skeleton = head_nets[1].meta.skeleton
        # if dense_connections:
        #     field_config.confidence_scales = (
        #         [1.0 for _ in skeleton] +
        #         [dense_coupling for _ in head_nets[2].meta.skeleton]
        #     )
        #     skeleton = skeleton + head_nets[2].meta.skeleton


factory = Factory.__call__
