from io import BytesIO

import math
import numpy as np
import logging
import base64
import random
import jsonlines

import torch
from fairseq.data import data_utils
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id[sort_order]
    src_tokens = src_tokens.index_select(0, sort_order)

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_images = patch_images.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target", left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge("target", left_pad=left_pad_target, move_eos_to_beginning=True)
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    return batch


class ConvertRGB:
    def __index__(self):
        pass

    def __call__(self, img):
        return img.convert("RGB")


class CaptionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        caption_path,
        image_path,
        tokenizer,
        src_dict,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        max_src_length=2,
        max_tgt_length=30,
        image_size=224,
        input_feeding=True
    ):
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1

        self.caption_dict = dict()
        with jsonlines.open(caption_path) as reader:
            for item in reader:
                self.caption_dict[item['image_id']] = item['text']
        self.slice_chunk = math.ceil(len(self.caption_dict) // self.slice_count)
        self.start_pos = self.slice_chunk * self.slice_id
        self.end_pos = min(self.slice_chunk * (self.slice_id + 1), len(self.caption_dict))
        self.row_count = self.end_pos - self.start_pos

        self.image_path = image_path
        self.tokenizer = tokenizer
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.eos = src_dict.eos()
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.image_size = image_size
        self.input_feeding = input_feeding

        self.bos_item = torch.LongTensor([self.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.src_dict.eos()])

        self.patch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            # lambda image: image.convert("RGB"),
            ConvertRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def encode_text(self, text, length=None):
        tokens_list = self.tokenizer.tokenize(text)
        tokens = ' '.join(tokens_list)
        s = self.tgt_dict.encode_line(
            line=tokens,
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        return s

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_chunk = math.ceil(self.row_count / num_workers)
        start = self.start_pos + worker_chunk * worker_id
        end = min(self.start_pos + worker_chunk * (worker_id + 1), len(self.caption_dict))
        with open(self.image_path) as f:
            for i, row in enumerate(f):
                if i < start:
                    continue
                if i >= end:
                    break
                image_id, image = row.strip().split('\t')
                try:
                    caption = random.choice(self.caption_dict[image_id])
                except Exception as e:
                    caption = ''
                yield self.process_data(image_id, image, caption)

    def process_data(self, image_id, image, caption):
        image_array = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_transform(image_array)

        src_item = torch.cat([self.bos_item, self.eos_item])
        caption_item = self.encode_text(' {}'.format(caption), length=self.max_tgt_length)
        target_item = torch.cat([caption_item, self.eos_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "target": target_item,
        }
        return example

    def __len__(self):
        return self.row_count

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch. """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return 64

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 128