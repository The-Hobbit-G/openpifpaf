from typing import List

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        for p in self.preprocess_list:
            if p is None:
                continue
            if (self.preprocess_list.index(p) == len(self.preprocess_list)-1) and (type(p) == list):
                process_results = [p_single(*args) for p_single in p]
                images = process_results[0][0]
                #encoders won't do any manipulation to the image, so they are always the same
                anns = []
                meta = []
                for result in process_results:
                    print('ann size: {} {}'.format(result[1][0].size())),result[1][1].size()
                    assert(len(result)==3)
                    anns.append(result[1])
                    meta.append(result[2])
                args = [images, anns, meta] #images will be the same as before, anns and meta would be type<list>
            else:
                args = p(*args)

        return args

    #enable us to further manipulate the preprocess compose after construction
    def pop(self, position):
        return self.preprocess_list.pop(position)

    def append(self, process):
        self.preprocess_list.append(process)
