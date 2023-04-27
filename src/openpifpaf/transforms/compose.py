from typing import List

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        anns = []
        for p in self.preprocess_list:
            if p is None:
                continue
            if (self.preprocess_list.index(p) == len(self.preprocess_list)-1) and (type(p) == list):
                images = None
                meta = None
                for p_single in p:
                    result = p_single(*args)
                    if images is None:
                        images = result[0]
                    if meta is None:
                        meta = result[2]
                    anns.append(result[1])
                args = images, anns, meta #images will be the same as before, anns and meta would be type<list>
                del images, anns, meta, result
            else:
                args = p(*args)

        return args

    #enable us to further manipulate the preprocess compose after construction
    def pop(self, position):
        return self.preprocess_list.pop(position)

    def append(self, process):
        self.preprocess_list.append(process)



'''
#last version
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
                meta = process_results[0][2]
                for result in process_results:
                    # print('ann size: {} {}'.format(result[1][0].size(),result[1][1].size()))
                    assert(len(result)==3)
                    anns.append(result[1])
                    # meta.append(result[2])
                args = images, anns, meta #images will be the same as before, anns and meta would be type<list>
                del process_results, images, anns, meta
            else:
                args = p(*args)

        return args

    #enable us to further manipulate the preprocess compose after construction
    def pop(self, position):
        return self.preprocess_list.pop(position)

    def append(self, process):
        self.preprocess_list.append(process)
'''



'''
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
                # print('ann size: {} {}'.format(result[1][0].size(),result[1][1].size()))
                assert(len(result)==3)
                anns.append(result[1])
                meta.append(result[2])
            args = [images, anns, meta] #images will be the same as before, anns and meta would be type<list>
        else:
            temp_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    temp_args.append(arg.detach())
                else:
                    temp_args.append(arg)
            args = p(*temp_args)
            # Delete temporary variables to free up memory
            del temp_args

    # Delete intermediate variables to free up memory
    del process_results, anns, meta

    return args


from typing import List
import contextlib

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        with contextlib.ExitStack() as stack:
            for p in self.preprocess_list:
                if p is None:
                    continue
                if (self.preprocess_list.index(p) == len(self.preprocess_list)-1) and (type(p) == list):
                    process_results = [p_single(*args) for p_single in p]
                    images = process_results[0][0]
                    anns = []
                    meta = []
                    for result in process_results:
                        assert(len(result)==3)
                        anns.append(result[1])
                        meta.append(result[2])
                    args = images, anns, meta
                else:
                    args = p(*args)
                # Delete images, anns, and meta after each iteration
                stack.callback(lambda: del images, anns, meta)
        return args

'''