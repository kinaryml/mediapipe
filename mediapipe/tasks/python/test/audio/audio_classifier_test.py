# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Tests for audio classifier."""

import enum
import os
from scipy.io import wavfile

from absl.testing import absltest
from absl.testing import parameterized

from mediapipe.tasks.python.components.containers import category
from mediapipe.tasks.python.components.containers import audio_data
from mediapipe.tasks.python.components.containers import classifications as classifications_module
from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.test import test_utils
from mediapipe.tasks.python.audio import audio_classifier
from mediapipe.tasks.python.audio.core import audio_task_running_mode

_BaseOptions = base_options_module.BaseOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_Category = category.Category
_ClassificationEntry = classifications_module.ClassificationEntry
_Classifications = classifications_module.Classifications
_ClassificationResult = classifications_module.ClassificationResult
_AudioData = audio_data.AudioData
_AudioClassifier = audio_classifier.AudioClassifier
_AudioClassifierOptions = audio_classifier.AudioClassifierOptions
_RUNNING_MODE = audio_task_running_mode.AudioTaskRunningMode

_YAMNET_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'
_TWO_HEADS_MODEL_FILE = 'two_heads.tflite'
_16K_WAVE_FILE = 'speech_16000_hz_mono.wav'
_48K_WAVE_FILE = 'speech_48000_hz_mono.wav'
_16K_WAVE_FILE_FOR_TWO_HEADS = 'two_heads_16000_hz_mono.wav'
_44K_WAVE_FILE_FOR_TWO_HEADS = 'two_heads_44100_hz_mono.wav'
_YAMNET_NUM_SAMPLES = 15600
_TEST_DATA_DIR = 'mediapipe/tasks/testdata/audio'


def _generate_empty_results(timestamp_ms: int) -> _ClassificationResult:
  return _ClassificationResult(classifications=[
      _Classifications(
          entries=[
              _ClassificationEntry(categories=[], timestamp_ms=timestamp_ms)
          ],
          head_index=0,
          head_name='probability')
  ])


def _generate_speech_results(timestamp_ms: int) -> _ClassificationResult:
  return _ClassificationResult(classifications=[
    _Classifications(
      entries=[
        _ClassificationEntry(categories=[], timestamp_ms=timestamp_ms)
      ],
      head_index=0,
      head_name='probability')
  ])


def _generate_two_heads_results(timestamp_ms: int) -> _ClassificationResult:
  return _ClassificationResult(classifications=[
    _Classifications(
      entries=[
        _ClassificationEntry(categories=[], timestamp_ms=timestamp_ms)
      ],
      head_index=0,
      head_name='probability')
  ])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioClassifierTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_sample_rate, self.test_wav_data = wavfile.read(
        test_utils.get_test_data_path(
            os.path.join(_TEST_DATA_DIR, _16K_WAVE_FILE)),
        True)
    # self.test_audio_clip = _Matrix.create_from_file(
    #     test_utils.get_test_data_path(
    #         os.path.join(_TEST_DATA_DIR, _16K_WAVE_FILE)))
    self.model_path = test_utils.get_test_data_path(
        os.path.join(_TEST_DATA_DIR, _YAMNET_MODEL_FILE))

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    with _AudioClassifier.create_from_model_path(self.model_path) as classifier:
      self.assertIsInstance(classifier, _AudioClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(model_asset_path=self.model_path)
    options = _AudioClassifierOptions(base_options=base_options)
    with _AudioClassifier.create_from_options(options) as classifier:
      self.assertIsInstance(classifier, _AudioClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(model_asset_path='')
      options = _AudioClassifierOptions(base_options=base_options)
      _AudioClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(model_asset_buffer=f.read())
      options = _AudioClassifierOptions(base_options=base_options)
      classifier = _AudioClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _AudioClassifier)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, 4, _generate_speech_results(0)),
      (ModelFileType.FILE_CONTENT, 4, _generate_speech_results(0)))
  def test_classify(self, model_file_type, max_results,
                    expected_classification_result):
    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(model_asset_path=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(model_asset_buffer=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    custom_classifier_options = _ClassifierOptions(max_results=max_results)
    options = _AudioClassifierOptions(
        base_options=base_options, classifier_options=custom_classifier_options)
    classifier = _AudioClassifier.create_from_options(options)

    # Performs audio classification on the input.
    audio_clip = _AudioData.create_from_array(self.test_wav_data,
                                              sample_rate=self.test_sample_rate)
    audio_result = classifier.classify(audio_clip, 16000)
    # Closes the classifier explicitly when the classifier is not used in
    # a context.
    classifier.close()


if __name__ == '__main__':
  absltest.main()
