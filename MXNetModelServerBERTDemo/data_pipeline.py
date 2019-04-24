# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
import unicodedata

import collections


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = ' '.join(tokenizer(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadExample(object):
    """A single training/test example for SQuAD question.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 example_id,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.example_id = example_id


class SquadFeature(object):
    """Single feature of a single example transform of the SQuAD question.

    """

    def __init__(self,
                 example_id,
                 qas_id,
                 doc_tokens,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 valid_length,
                 segment_ids,
                 start_position,
                 end_position,
                 is_impossible):
        self.example_id = example_id
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.valid_length = valid_length
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SQuADTransform(object):
    """Dataset Transformation for BERT-style QA.

    The transformation is processed in the following steps:
    - Convert from gluonnlp.data.SQuAD's record to SquadExample.
    - Tokenize the question_text in the example.
    - For examples where the document is too long,
      use a sliding window to split into multiple features and
      record whether each token is a maximum context.
    - Tokenize the split document chunks.
    - Combine the token of question_text with the token
      of the document and insert [CLS] and [SEP].
    - Generate the start position and end position of the answer.
    - Generate valid length.

    E.g:

    Inputs:
        question_text: 'When did BBC Japan begin broadcasting?'
        doc_tokens: ['BBC','Japan','was','a','general','entertainment','channel,',
                    'which','operated','between','December','2004','and','April',
                    '2006.','It','ceased','operations','after','its','Japanese',
                    'distributor','folded.']
        start_position: 10
        end_position: 11
        orig_answer_text: 'December 2004'
    Processed:
        tokens: ['[CLS]','when','did','bbc','japan','begin','broadcasting','?',
                '[SEP]','bbc','japan','was','a','general','entertainment','channel',
                ',','which','operated','between','december','2004','and','april',
                '2006','.','it','ceased','operations','after','its','japanese',
                'distributor','folded','.','[SEP]']
        segment_ids: [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        start_position: 20
        end_position: 21
        valid_length: 36

    Because of the sliding window approach taken to scoring documents, a single
    token can appear in multiple documents.
    So you need to record whether each token is a maximum context. E.g.
       Doc: the man went to the store and bought a gallon of milk
       Span A: the man went to the
       Span B: to the store and bought
       Span C: and bought a gallon of
       ...

    Now the word 'bought' will have two scores from spans B and C. We only
    want to consider the score with "maximum context", which we define as
    the *minimum* of its left and right context (the *sum* of left and
    right context will always be the same, of course).

    In the example the maximum context for 'bought' would be span C since
    it has 1 left context and 3 right context, while span B has 4 left context
    and 0 right context.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int, default 384
        Maximum sequence length of the sentences.
    doc_stride : int, default 128
        When splitting up a long document into chunks,
        how much stride to take between chunks.
    max_query_length : int, default 64
        The maximum length of the query tokens.
    is_training : bool, default True
        Whether to run training.
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 is_training=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.is_training = is_training

    def _is_whitespace(self, c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(
                c) == 0x202F:
            return True
        return False

    def _to_squad_example(self, record):
        example_id = record[0]
        qas_id = record[1]
        question_text = record[2]
        paragraph_text = record[3]
        orig_answer_text = record[4][0]
        answer_offset = record[5][0]
        is_impossible = record[6] if len(record) == 7 else False

        doc_tokens = []

        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if self._is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        start_position = None
        end_position = None

        if self.is_training:
            if not is_impossible:
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[
                    answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = ' '.join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = ' '.join(
                    _whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    print('Could not find answer: %s vs. %s' %
                          (actual_text, cleaned_answer_text))
                    return None
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ''

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            example_id=example_id,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        return example

    def _transform(self, *record):
        example = self._to_squad_example(record)
        if not example:
            return None

        features = []
        query_tokens = self.tokenizer(example.question_text)

        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[0:self.max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if self.is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if self.is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                self.tokenizer, example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            'DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append('[CLS]')
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append('[SEP]')
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            valid_length = len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(self.tokenizer.vocab['<PAD>'])
                segment_ids.append(self.tokenizer.vocab['<PAD>'])

            assert len(input_ids) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            start_position = 0
            end_position = 0
            if self.is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if self.is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            features.append(SquadFeature(example_id=example.example_id,
                                         qas_id=example.qas_id,
                                         doc_tokens=example.doc_tokens,
                                         doc_span_index=doc_span_index,
                                         tokens=tokens,
                                         token_to_orig_map=token_to_orig_map,
                                         token_is_max_context=token_is_max_context,
                                         input_ids=input_ids,
                                         valid_length=valid_length,
                                         segment_ids=segment_ids,
                                         start_position=start_position,
                                         end_position=end_position,
                                         is_impossible=example.is_impossible))
        return features

    def __call__(self, record):
        examples = self._transform(*record)
        if not examples:
            return None
        features = []

        for _example in examples:
            feature = [_example.example_id, _example.input_ids, _example.segment_ids,
                       _example.valid_length, _example.start_position, _example.end_position]
            features.append(feature)

        return examples, features


class BasicTokenizer():
    r"""Runs basic tokenization

    performs invalid character removal (e.g. control chars) and whitespace.
    tokenize CJK chars.
    splits punctuation on a piece of text.
    strips accents and convert to lower case.(If lower_case is true)

    Parameters
    ----------
    lower_case : bool, default True
        whether the text strips accents and convert to lower case.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.BasicTokenizer(lower_case=True)
    >>> tokenizer(u" \tHeLLo!how  \n Are yoU?  ")
    ['hello', '!', 'how', 'are', 'you', '?']
    >>> tokenizer = gluonnlp.data.BasicTokenizer(lower_case=False)
    >>> tokenizer(u" \tHeLLo!how  \n Are yoU?  ")
    ['HeLLo', '!', 'how', 'Are', 'yoU', '?']

    """

    def __init__(self, lower_case=True):
        self.lower_case = lower_case

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample:  str (unicode for Python 2)
            The string to tokenize. Must be unicode.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self._tokenize(sample)

    def _tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = _whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = _whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp in (0, 0xfffd) or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char in ['\t', '\n', '\r']:
            return False
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            return True
        return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF)
                or (0x3400 <= cp <= 0x4DBF)
                or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F)
                or (0x2B740 <= cp <= 0x2B81F)
                or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF)
                or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        group0 = 33 <= cp <= 47
        group1 = 58 <= cp <= 64
        group2 = 91 <= cp <= 96
        group3 = 123 <= cp <= 126

        if group0 or group1 or group2 or group3:
            return True

        cat = unicodedata.category(char)

        if cat.startswith('P'):
            return True

        return False

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char in [' ', '\t', '\n', '\r']:
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            return True
        return False


class BERTTokenizer(object):
    r"""End-to-end tokenization for BERT models.

    Parameters
    ----------
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary for the corpus.
    lower_case : bool, default True
        whether the text strips accents and convert to lower case.
        If you use the BERT pre-training model,
        lower_case is set to Flase when using the cased model,
        otherwise it is set to True.
    max_input_chars_per_word : int, default 200


    Examples
    --------
    >>> _,vocab = gluonnlp.model.bert_12_768_12(dataset_name='wiki_multilingual',pretrained=False)
    >>> tokenizer = gluonnlp.data.BERTTokenizer(vocab=vocab)
    >>> tokenizer(u"gluonnlp: 使NLP变得简单。")
    ['gl', '##uo', '##nn', '##lp', ':', '使', 'nl', '##p', '变', '得', '简', '单', '。']

    """

    def __init__(self, vocab, lower_case=True, max_input_chars_per_word=200):
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word
        self.basic_tokenizer = BasicTokenizer(lower_case=lower_case)

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str (unicode for Python 2)
            The string to tokenize. Must be unicode.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self._tokenizer(sample)

    def _tokenizer(self, text):
        split_tokens = []
        for token in self.basic_tokenizer(text):
            for sub_token in self._tokenize_wordpiece(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _tokenize_wordpiece(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in _whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.vocab.unknown_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.vocab.unknown_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        return self.vocab.to_indices(tokens)
