from .preprocess import Preprocess


class Encoders(Preprocess):
    """Preprocess operation that runs encoders."""
    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, image, anns, meta):
        anns = [enc(image, anns, meta) for enc in self.encoders]
        #in case of using cifcafdet, anns would be [[cif1,cif2,...], [caf1,caf2,...]]
        
        cif_detections = anns[0][1]
        anns = [anns[0][0], anns[1]]
        meta['cif_detections'] = cif_detections


        # for ann in anns:
        #     print('ann size {}'.format(ann.size()))
        # for enc in self.encoders:
        #     print(enc.meta.stride)
        meta['head_indices'] = [enc.meta.head_index for enc in self.encoders]
        return image, anns, meta
