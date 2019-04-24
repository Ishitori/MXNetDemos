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
import collections
from collections import namedtuple, OrderedDict
import six

from mxnet import nd

Result = collections.namedtuple(
    '_Result', ['start_logits', 'end_logits'])


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i, _ in enumerate(index_and_score):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, tokenizer):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = OrderedDict()
        for (i, c) in enumerate(text):
            if c == ' ':
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = ''.join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tok_text = ' '.join(tokenizer(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def predictions(dev_dataset,
                all_results,
                tokenizer,
                max_answer_length=64,
                null_score_diff_threshold=0.0,
                n_best_size=10,
                version_2=False):
    """Get prediction results

    Parameters
    ----------
    dev_dataset: dataset
        Examples of transform.
    all_results: dict
        A dictionary containing model prediction results.
    tokenizer: callable
        Tokenizer function.
    max_answer_length: int, default 64
        Maximum length of the answer tokens.
    null_score_diff_threshold: float, default 0.0
        If null_score - best_non_null is greater than the threshold predict null.
    n_best_size: int, default 10
        The total number of n-best predictions.
    version_2: bool, default False
        If true, the SQuAD examples contain some that do not have an answer.
    Returns
    -------
    all_predictions: dict
        All final predictions.
    all_nbest_json: dict
        All n-best predictions.
    scores_diff_json: dict
        If version_2 is True.
        Difference between the null score and the score of best non-null.
    """
    # score_null = 1000000  # large and positive
    # min_null_feature_index = 0  # the paragraph slice with min mull score
    # null_start_logit = 0  # the start logit at the slice with min null score
    # null_end_logit = 0  # the end logit at the slice with min null score
    max_answer_length = max_answer_length
    null_score_diff_threshold = null_score_diff_threshold

    _PrelimPrediction = namedtuple('PrelimPrediction',
                                   ['feature_index', 'start_index', 'end_index',
                                    'start_logit', 'end_logit'])

    _NbestPrediction = namedtuple(
        'NbestPrediction', ['text', 'start_logit', 'end_logit'])

    all_predictions = OrderedDict()
    all_nbest_json = OrderedDict()
    scores_diff_json = OrderedDict()

    example_qas_id = 0
    prelim_predictions = []

    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score

    for features_id, (result, feature) in enumerate(zip(all_results, dev_dataset)):
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)

        if version_2:
            feature_null_score = result.start_logits[0] + \
                result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = features_id
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=features_id,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index]))

    if version_2:
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = dev_dataset[pred.feature_index]
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(
                pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = feature.doc_tokens[orig_doc_start:(
                orig_doc_end + 1)]
            tok_text = ' '.join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(' ##', '')
            tok_text = tok_text.replace('##', '')

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = ' '.join(tok_text.split())
            orig_text = ' '.join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ''
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if version_2:
        if '' not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text='',
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text='empty', start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = nd.softmax(nd.array(total_scores)).asnumpy()

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = OrderedDict()
        output['text'] = entry.text
        output['probability'] = float(probs[i])
        output['start_logit'] = entry.start_logit
        output['end_logit'] = entry.end_logit
        nbest_json.append(output)

    if not version_2:
        all_predictions[example_qas_id] = nbest_json[0]['text']
        scores_diff_json = None
    else:
        # predict '' iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.start_logit - \
            best_non_null_entry.end_logit

        scores_diff_json[example_qas_id] = score_diff
        if score_diff > null_score_diff_threshold:
            all_predictions[example_qas_id] = ''
        else:
            all_predictions[example_qas_id] = best_non_null_entry.text

    all_nbest_json[example_qas_id] = nbest_json
    return all_predictions, all_nbest_json, scores_diff_json
