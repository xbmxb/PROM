# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

from .tokenization_roberta import RobertaTokenizer
from .tokenization_utils_base import BatchEncoding, TruncationStrategy, PaddingStrategy, TensorType
from .utils import logging
import warnings


logger = logging.get_logger(__name__)


# vocab and merges same as roberta
vocab_url = "https://huggingface.co/roberta-large/resolve/main/vocab.json"
merges_url = "https://huggingface.co/roberta-large/resolve/main/merges.txt"
_all_bart_models = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "yjernite/bart_eli5",
    # This is not exhaustive: see https://huggingface.co/models?filter=bart
]


class BartTokenizer(RobertaTokenizer):
    r"""
    Construct a BART tokenizer.

    :class:`~transformers.BartTokenizer` is identical to :class:`~transformers.RobertaTokenizer` and adds a new
    :meth:`~transformers.BartTokenizer.prepare_seq2seq_batch`

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning the
    initialization parameters and other methods.
    """
    # merges and vocab same as Roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
    }

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = "None",
        truncation=True,
        **kwargs,
    ) -> BatchEncoding:
        r"""

        Prepare a batch that can be passed directly to an instance of :class:`~transformers.BartModel`.

        Args:
            src_texts: (:obj:`List[str]`):
                List of documents to summarize or source language texts.
            tgt_texts: (:obj:`List[str]`, `optional`):
                List of summaries or target language texts.
            max_length (:obj:`int`, `optional`):
                Controls the maximum length for encoder inputs (documents to summarize or source language texts). If
                left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            max_target_length (:obj:`int`, `optional`):
                Controls the maximum length of decoder inputs (target language texts or summaries). If left unset or
                set to :obj:`None`, this will use the max_length value.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`True`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            **kwargs:
                Additional keyword arguments passed along to :obj:`self.__call__`.

        Returns:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to the encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **labels** -- List of token ids for tgt_texts

            The full set of keys ``[input_ids, attention_mask, labels]``, will only be returned if tgt_texts is passed.
            Otherwise, input_ids, attention_mask will be the only keys.
        """
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        if max_length is None:
            max_length = self.model_max_length
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs


class MyBartTokenizer(BartTokenizer):
    TextInput = str
    PreTokenizedInput = List[str]
    EncodedInput = List[int]
    TextInputPair = Tuple[str, str]
    PreTokenizedInputPair = Tuple[List[str], List[str]]
    EncodedInputPair = Tuple[List[int], List[int]]

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = "None",
        truncation=True,
        **kwargs,
    ) -> BatchEncoding:
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        if max_length is None:
            max_length = self.model_max_length
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs  

    def __call__(
        self, 
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], 
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None, 
        add_special_tokens: bool = True, 
        padding: Union[bool, str, PaddingStrategy] = False, 
        truncation: Union[bool, str, TruncationStrategy] = False, 
        max_length: Optional[int] = None, 
        stride: int = 0, 
        is_split_into_words: bool = False, 
        pad_to_multiple_of: Optional[int] = None, 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        return_token_type_ids: Optional[bool] = None, 
        return_attention_mask: Optional[bool] = None, 
        return_overflowing_tokens: bool = False, 
        return_special_tokens_mask: bool = False, 
        return_offsets_mapping: bool = False, 
        return_length: bool = False, 
        verbose: bool = True, **kwargs
    ) -> BatchEncoding:
        batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
        # return self.batch_encode_plus(
        #     batch_text_or_text_pairs=batch_text_or_text_pairs,
        #     add_special_tokens=add_special_tokens,
        #     padding=padding,
        #     truncation=truncation,
        #     max_length=max_length,
        #     stride=stride,
        #     is_split_into_words=is_split_into_words,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     return_tensors=return_tensors,
        #     return_token_type_ids=return_token_type_ids,
        #     return_attention_mask=return_attention_mask,
        #     return_overflowing_tokens=return_overflowing_tokens,
        #     return_special_tokens_mask=return_special_tokens_mask,
        #     return_offsets_mapping=return_offsets_mapping,
        #     return_length=return_length,
        #     verbose=verbose,
        #     **kwargs,
        # )
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            # elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
            #     if is_split_into_words:
            #         tokens = list(
            #             itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
            #         )
            #         return self.convert_tokens_to_ids(tokens)
            #     else:
            #         return self.convert_tokens_to_ids(text)
            # elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
            #     return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )
        if "is_pretokenized" in kwargs:
            warnings.warn(
                "`is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.",
                FutureWarning,
            )
            is_split_into_words = kwargs.pop("is_pretokenized")

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            # if not isinstance(ids_or_pair_ids, (list, tuple)):
            #     ids, pair_ids = ids_or_pair_ids, None
            # elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
            #     ids, pair_ids = ids_or_pair_ids, None
            # else:
            #     ids, pair_ids = ids_or_pair_ids
            ids, pair_ids = ids_or_pair_ids, None
            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))
        batch_ids_pairs = input_ids
        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            ids, pair_ids = first_ids, second_ids
            # if "return_lengths" in kwargs:
            #     if verbose:
            #         warnings.warn(
            #             "The PreTrainedTokenizerBase.prepare_for_model `return_lengths` parameter is deprecated. "
            #             "Please use `return_length` instead.",
            #             FutureWarning,
            #         )
            #     return_length = kwargs["return_lengths"]

            # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
            padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                verbose=verbose,
                **kwargs,
            )

            pair = bool(pair_ids is not None)
            len_ids = len(ids)
            len_pair_ids = len(pair_ids) if pair else 0

            if return_token_type_ids is not None and not add_special_tokens:
                raise ValueError(
                    "Asking to return token_type_ids while setting add_special_tokens to False "
                    "results in an undefined behavior. Please set add_special_tokens to True or "
                    "set return_token_type_ids to None."
                )

            # Load from model defaults
            if return_token_type_ids is None:
                return_token_type_ids = "token_type_ids" in self.model_input_names
            if return_attention_mask is None:
                return_attention_mask = "attention_mask" in self.model_input_names

            encoded_inputs = {}

            # Compute the total size of the returned encodings
            total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

            # Truncation: Handle max sequence length
            overflowing_tokens = []
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
                ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                    ids,
                    pair_ids=pair_ids,
                    num_tokens_to_remove=total_len - max_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )

            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

            # Add special tokens
            if add_special_tokens:
                sequence = self.build_inputs_with_special_tokens(ids, pair_ids) # move here
                token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids) # move here
            else:
                sequence = ids + pair_ids if pair else ids
                token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

            # Build output dictionary
            encoded_inputs["input_ids"] = sequence
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = token_type_ids
            if return_special_tokens_mask:
                if add_special_tokens:
                    encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
                else:
                    encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

            # Check lengths
            if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length and verbose:
                if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                    logger.warning(
                        "Token indices sequence length is longer than the specified maximum sequence length "
                        "for this model ({} > {}). Running this sequence through the model will result in "
                        "indexing errors".format(len(encoded_inputs["input_ids"]), self.model_max_length)
                    )
                self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
            # # Padding
            # if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            #     encoded_inputs = self.pad(
            #         encoded_inputs,
            #         max_length=max_length,
            #         padding=padding_strategy.value,
            #         pad_to_multiple_of=pad_to_multiple_of,
            #         return_attention_mask=return_attention_mask,
            #     )
            if return_length:
                encoded_inputs["length"] = len(encoded_inputs["input_ids"])

            outputs = BatchEncoding(
                encoded_inputs, tensor_type=None, prepend_batch_axis=False
            )
        # # return batch_outputs
        #     outputs = self.prepare_for_model(
        #         first_ids,
        #         second_ids,
        #         add_special_tokens=add_special_tokens,
        #         padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
        #         truncation=truncation_strategy.value,
        #         max_length=max_length,
        #         stride=stride,
        #         pad_to_multiple_of=None,  # we pad in batch afterward
        #         return_attention_mask=False,  # we pad in batch afterward
        #         return_token_type_ids=return_token_type_ids,
        #         return_overflowing_tokens=return_overflowing_tokens,
        #         return_special_tokens_mask=return_special_tokens_mask,
        #         return_length=return_length,
        #         return_tensors=None,  # We convert the whole batch to tensors at the end
        #         prepend_batch_axis=False,
        #         verbose=verbose,
        #     )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # batch_outputs = self.pad(
        #     batch_outputs,
        #     padding=padding_strategy.value,
        #     max_length=max_length,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     return_attention_mask=return_attention_mask,
        # )
        encoded_inputs = batch_outputs
        
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
        assert "input_ids" in encoded_inputs, (
            "You should supply an encoding or a list of encodings to this method. "
            "An encoding is the output of one the encoding methods of the tokenizer, i.e. "
            "__call__/encode_plus/batch_encode_plus. "
        )
        # if not encoded_inputs["input_ids"]:
        #     if return_attention_mask:
        #         encoded_inputs["attention_mask"] = []
        #     return encoded_inputs
        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch
        # first_element = encoded_inputs["input_ids"][0]
        # if isinstance(first_element, (list, tuple)) and first_element:
        #     first_element = first_element[0]
        # if not isinstance(first_element, int):
        #     if is_tf_available() and isinstance(first_element, tf.Tensor):
        #         return_tensors = "tf" if return_tensors is None else return_tensors
        #     elif is_torch_available() and isinstance(first_element, torch.Tensor):
        #         return_tensors = "pt" if return_tensors is None else return_tensors
        #     elif isinstance(first_element, np.ndarray):
        #         return_tensors = "np" if return_tensors is None else return_tensors
        #     else:
        #         raise ValueError(
        #             f"type of {first_element} unknown: {type(first_element)}. "
        #             f"Should be one of a python, numpy, pytorch or tensorflow object."
        #         )

        #     for key, value in encoded_inputs.items():
        #         encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        # if encoded_inputs["input_ids"] and not isinstance(encoded_inputs["input_ids"][0], (list, tuple)):
        #     encoded_inputs = self._pad(
        #         encoded_inputs,
        #         max_length=max_length,
        #         padding_strategy=padding_strategy,
        #         pad_to_multiple_of=pad_to_multiple_of,
        #         return_attention_mask=return_attention_mask,
        #     )
        #     return BatchEncoding(encoded_inputs, tensor_type=return_tensors)
        batch_size = len(encoded_inputs["input_ids"])
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in encoded_inputs["input_ids"])
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            encoded_input = dict((k, v[i]) for k, v in encoded_inputs.items())
            # outputs = self._pad(
            #     inputs,
            #     max_length=max_length,
            #     padding_strategy=padding_strategy,
            #     pad_to_multiple_of=pad_to_multiple_of,
            #     return_attention_mask=return_attention_mask,
            # )
            # Load from model defaults
            '''
            - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
            - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
            - PaddingStrategy.DO_NOT_PAD: Do not pad
            '''
            if return_attention_mask is None:
                return_attention_mask = "attention_mask" in self.model_input_names

            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(encoded_input["input_ids"])

            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_input["input_ids"]) != max_length
            )

            if needs_to_be_padded:
                difference = max_length - len(encoded_input["input_ids"])
                if self.padding_side == "right":
                    if return_attention_mask:
                        encoded_input["attention_mask"] = [1] * len(encoded_input["input_ids"]) + [0] * difference
                    if "token_type_ids" in encoded_input:
                        encoded_input["token_type_ids"] = (
                            encoded_input["token_type_ids"] + [self.pad_token_type_id] * difference
                        )
                    if "special_tokens_mask" in encoded_input:
                        encoded_input["special_tokens_mask"] = encoded_input["special_tokens_mask"] + [1] * difference
                    encoded_input["input_ids"] = encoded_input["input_ids"] + [self.pad_token_id] * difference # tensor or list
                elif self.padding_side == "left":
                    if return_attention_mask:
                        encoded_input["attention_mask"] = [0] * difference + [1] * len(encoded_input["input_ids"])
                    if "token_type_ids" in encoded_input:
                        encoded_input["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_input[
                            "token_type_ids"
                        ]
                    if "special_tokens_mask" in encoded_input:
                        encoded_input["special_tokens_mask"] = [1] * difference + encoded_input["special_tokens_mask"]
                    encoded_input["input_ids"] = [self.pad_token_id] * difference + encoded_input["input_ids"]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))
            else:
                if return_attention_mask:
                    encoded_input["attention_mask"] = [1] * len(encoded_input["input_ids"])

            outputs =  encoded_input

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # return batch_outputs

        # batch_outputs = self._batch_prepare_for_model(
        #     input_ids,
        #     add_special_tokens=add_special_tokens,
        #     padding_strategy=padding_strategy,
        #     truncation_strategy=truncation_strategy,
        #     max_length=max_length,
        #     stride=stride,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     return_attention_mask=return_attention_mask,
        #     return_token_type_ids=return_token_type_ids,
        #     return_overflowing_tokens=return_overflowing_tokens,
        #     return_special_tokens_mask=return_special_tokens_mask,
        #     return_length=return_length,
        #     return_tensors=return_tensors,
        #     verbose=verbose,
        # )

        # return BatchEncoding(batch_outputs)

        # return self._batch_encode_plus(
        #     batch_text_or_text_pairs=batch_text_or_text_pairs,
        #     add_special_tokens=add_special_tokens,
        #     padding_strategy=padding_strategy,
        #     truncation_strategy=truncation_strategy,
        #     max_length=max_length,
        #     stride=stride,
        #     is_split_into_words=is_split_into_words,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     return_tensors=return_tensors,
        #     return_token_type_ids=return_token_type_ids,
        #     return_attention_mask=return_attention_mask,
        #     return_overflowing_tokens=return_overflowing_tokens,
        #     return_special_tokens_mask=return_special_tokens_mask,
        #     return_offsets_mapping=return_offsets_mapping,
        #     return_length=return_length,
        #     verbose=verbose,
        #     **kwargs,
        # )
        # return super().__call__(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs) 
