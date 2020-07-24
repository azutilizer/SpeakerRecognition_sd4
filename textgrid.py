import os
import re
import json
from collections import OrderedDict


def remove_empty_lines(text_list):
    """remove empty lines"""
    assert len(text_list) > 0
    assert isinstance(text_list, list)
    text_list = [t.strip() for t in text_list]
    if b'' in text_list:
        text_list.remove(b'')
    return text_list


class TextGrid(object):
    """
    Tier 1: "Bookmark"
    Tier 2: "SpeakerID"
    """
    def __init__(self):
        self.line_count = 0
        self.text = None
        self.size = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.tier_list = []

    def set_text(self, contents):
        self.text = contents
        self._get_type()
        self._get_time_intval()
        self._get_size()
        self.tier_list = []
        self._get_item_list()

    def _extract_pattern(self, pattern, inc):
        try:
            group = re.match(pattern, self.text[self.line_count].decode("utf-8")).group(1)
            self.line_count += inc
        except AttributeError:
            raise ValueError("File format error at line %d:%s" % (self.line_count, self.text[self.line_count]))
        return group

    def _get_type(self):
        self.file_type = self._extract_pattern(r"File type = \"(.*)\"", 2)

    def _get_time_intval(self):
        self.xmin = self._extract_pattern(r"xmin = (.*)", 1)
        self.xmax = self._extract_pattern(r"xmax = (.*)", 2)

    def _get_size(self):
        self.size = int(self._extract_pattern(r"size = (.*)", 2))

    def _get_item_list(self):
        """Only supports IntervalTier currently"""
        for itemIdx in range(1, self.size + 1):
            tier = OrderedDict()
            item_list = []
            tier_idx = self._extract_pattern(r"item \[(.*)\]:", 1)
            tier_class = self._extract_pattern(r"class = \"(.*)\"", 1)
            if tier_class != "IntervalTier":
                raise NotImplementedError("Only IntervalTier class is supported currently")
            tier_name = self._extract_pattern(r"name = \"(.*)\"", 1)
            tier_xmin = self._extract_pattern(r"xmin = (.*)", 1)
            tier_xmax = self._extract_pattern(r"xmax = (.*)", 1)
            tier_size = self._extract_pattern(r"intervals: size = (.*)", 1)
            for i in range(int(tier_size)):
                item = OrderedDict()
                item["idx"] = self._extract_pattern(r"intervals \[(.*)\]", 1)
                item["xmin"] = self._extract_pattern(r"xmin = (.*)", 1)
                item["xmax"] = self._extract_pattern(r"xmax = (.*)", 1)
                item["text"] = self._extract_pattern(r"text = \"(.*)\"", 1)
                item_list.append(item)
            tier["idx"] = tier_idx
            tier["class"] = tier_class
            tier["name"] = tier_name
            tier["xmin"] = tier_xmin
            tier["xmax"] = tier_xmax
            tier["size"] = tier_size
            tier["items"] = item_list
            self.tier_list.append(tier)

    def toJson(self):
        _json = OrderedDict()
        _json["file_type"] = self.file_type
        _json["xmin"] = self.xmin
        _json["xmax"] = self.xmax
        _json["size"] = self.size
        _json["tiers"] = self.tier_list
        return json.dumps(_json, ensure_ascii=False, indent=2).encode("utf-8")

    def to_TextGrid(self, dst_file):
        assert str(dst_file).endswith('.TextGrid')
        with open(dst_file, 'wt') as f:
            f.write("File type = \"ooTextFile\"\n")
            f.write("Object class = \"TextGrid\"\n\n")

            f.write("xmin = {}\n".format(self.xmin))
            f.write("xmax = {}\n".format(self.xmax))
            f.write("tiers? <exists>\n")
            f.write("size = {}\n".format(self.size))
            f.write("item []:\n")

            for tier in self.tier_list:
                f.write("\titem [{}]:\n".format(tier['idx']))
                f.write("\t\tclass = \"{}\"\n".format(tier["class"]))
                f.write("\t\tname = \"{}\"\n".format(tier["name"]))
                f.write("\t\txmin = {}\n".format(tier['xmin']))
                f.write("\t\txmax = {}\n".format(tier['xmax']))
                f.write("\t\tintervals: size = {}\n".format(tier["size"]))
                for interv in tier['items']:
                    f.write("\t\tintervals [{}]:\n".format(interv["idx"]))
                    f.write("\t\t\txmin = {:.3f}\n".format(interv["xmin"]))
                    f.write("\t\t\txmax = {:.3f}\n".format(interv["xmax"]))
                    f.write("\t\t\ttext = \"{}\"\n".format(interv["text"]))

    def get_bookmark_list(self):
        for tier in self.tier_list:
            if tier['name'] != 'Bookmark':
                continue
            result = []
            for seg in tier['items']:
                xmin = float(seg['xmin'])
                xmax = float(seg['xmax'])
                word = seg['text']
                if word in ['sil', 'sp', '']:
                    continue
                result.append([xmin, xmax, word])
            return result

    def get_speaker_list(self):
        for tier in self.tier_list:
            if tier['name'] != 'SpeakerID':
                continue
            result = []
            for seg in tier['items']:
                xmin = float(seg['xmin'])
                xmax = float(seg['xmax'])
                word = seg['text']
                if word == '':
                    continue
                result.append([xmin, xmax, word])
            return result

    def add_tier_list(self, item_name, items):
        """
        :param item_name: should be "Bookmark" or "SpeakerID" here.
        :param items: list of [[0.0, 3.8578, '1_noise'], [3.8578, 5.0524, '2_noise'],]
        :return:
        """
        tier_size = len(items)
        tier_class = "IntervalTier"
        item_list = []

        for i, row in enumerate(items):
            item = OrderedDict()
            item["idx"] = i + 1
            item["xmin"] = row[0]
            item["xmax"] = row[1]
            item["text"] = row[2]
            item_list.append(item)

        tier = OrderedDict()
        tier["idx"] = self.size + 1
        tier["class"] = tier_class
        tier["name"] = item_name
        tier["xmin"] = items[0][0]
        tier["xmax"] = items[-1][1]
        tier["size"] = tier_size
        tier["items"] = item_list
        self.tier_list.append(tier)

        if self.size == 0:
            self.xmin = tier["xmin"]
            self.xmax = tier["xmax"]
        else:
            self.xmin = min(self.xmin, tier["xmin"])
            self.xmax = max(self.xmax, tier["xmax"])

        self.size += 1


def read_grid_file(text_grid_file):
    if not os.path.exists(text_grid_file):
        raise IOError("input textgrid file can't be found")

    with open(text_grid_file, "rb") as f:
        text = f.readlines()
    if len(text) == 0:
        raise IOError("input textgrid file can't be empty")
    text = remove_empty_lines(text)
    textgrid = TextGrid()
    textgrid.set_text(text)
    wrd_list = textgrid.get_bookmark_list()
    return wrd_list


def test_write():
    tgt = TextGrid()
    tmp_list = [[0.0, 3.8578, '1_noise'], [3.8578, 5.0524, '2_noise'], [5.0524, 6.2989, '3_noise'],
                [6.2989, 7.4935, '4_noise']]
    tgt.add_tier_list("Bookmark", tmp_list)

    tmp_list = [[0.0, 3.8578, 'NULL'], [3.8578, 5.0524, 'CIND1001'], [5.0524, 6.2989, 'SIPLIND12231'],
                [6.2989, 7.4935, 'CIND1001']]
    tgt.add_tier_list("SpeakerID", tmp_list)

    tgt.to_TextGrid("test.TextGrid")


if __name__ == '__main__':
    # grid_file = os.path.join('..', '..', 'wav_textgrid', '1559884678_1.TextGrid')
    # print(read_grid_file(grid_file))

    test_write()
