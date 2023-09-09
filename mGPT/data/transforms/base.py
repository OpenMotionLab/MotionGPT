# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from dataclasses import dataclass, fields


class Transform:

    def collate(self, lst_datastruct):
        from ..tools import collate_tensor_with_padding
        example = lst_datastruct[0]

        def collate_or_none(key):
            if example[key] is None:
                return None
            key_lst = [x[key] for x in lst_datastruct]
            return collate_tensor_with_padding(key_lst)

        kwargs = {key: collate_or_none(key) for key in example.datakeys}

        return self.Datastruct(**kwargs)


# Inspired from SMPLX library
# need to define "datakeys" and transforms
@dataclass
class Datastruct:

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def to(self, *args, **kwargs):
        for key in self.datakeys:
            if self[key] is not None:
                self[key] = self[key].to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self[self.datakeys[0]].device

    def detach(self):

        def detach_or_none(tensor):
            if tensor is not None:
                return tensor.detach()
            return None

        kwargs = {key: detach_or_none(self[key]) for key in self.datakeys}
        return self.transforms.Datastruct(**kwargs)
