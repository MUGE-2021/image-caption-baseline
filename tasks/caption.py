import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from fairseq import utils
from fairseq.data import iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from data.caption_dataset import CaptionDataset
from utils.wordpiece import BertTokenizer

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@dataclass
class CaptionConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    dict_file: Optional[str] = field(
        default=None, metadata={"help": "chinese dict file"}
    )
    vocab_file: Optional[str] = field(
        default=None, metadata={"help": "chinese vocab file"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )

    image_size: int = field(
        default=224, metadata={"help": "image size"}
    )
    max_src_length: int = field(
        default=2, metadata={"help": "the maximum source sequence length"}
    )
    max_tgt_length: int = field(
        default=40, metadata={"help": "the maximum target sequence length"}
    )


@register_task("caption", dataclass=CaptionConfig)
class CaptionTask(FairseqTask):

    cfg: CaptionConfig

    def __init__(self, cfg: CaptionConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: CaptionConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(cfg.dict_file)
        tgt_dict = cls.load_dictionary(cfg.dict_file)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("src dictionary: {} types".format(len(src_dict)))
        logger.info("tgt dictionary: {} types".format(len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        data_path = self.cfg.data
        caption_path = os.path.join(data_path, 'IC_{}.jsonl'.format(split))
        image_path = os.path.join(data_path, 'IC_{}.tsv'.format(split))
        self.datasets[split] = CaptionDataset(
            caption_path,
            image_path,
            self.tokenizer,
            self.src_dict,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            image_size=self.cfg.image_size,
        )

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):

        # return a reusable, sharded iterator
        epoch_iter = iterators.StreamingEpochBatchIterator(
            dataset=dataset,
            max_sentences=max_sentences,
            collate_fn=dataset.collater,
            epoch=epoch,
            num_workers=num_workers,
            buffer_size=data_buffer_size,
            timeout=0
        )

        return epoch_iter

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.tokenizer = BertTokenizer(cfg.vocab_file)
        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict