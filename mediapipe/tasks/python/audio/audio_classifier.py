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
"""MediaPipe audio classifier task."""

import dataclasses
from typing import Callable, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
# TODO: Import MPImage directly one we have an alias
from mediapipe.python._framework_bindings import matrix
from mediapipe.python._framework_bindings import packet
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.cc.audio.audio_classifier.proto import audio_classifier_graph_options_pb2
from mediapipe.tasks.python.components.containers import classifications
from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.audio.core import base_audio_task_api
from mediapipe.tasks.python.audio.core import audio_task_running_mode

_BaseOptions = base_options_module.BaseOptions
_AudioClassifierGraphOptionsProto = audio_classifier_graph_options_pb2.AudioClassifierGraphOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_RunningMode = audio_task_running_mode.AudioTaskRunningMode
_TaskInfo = task_info_module.TaskInfo

_CLASSIFICATION_RESULT_OUT_STREAM_NAME = 'classification_result_out'
_CLASSIFICATION_RESULT_TAG = 'CLASSIFICATION_RESULT'
_AUDIO_IN_STREAM_NAME = 'audio_in'
_AUDIO_TAG = 'AUDIO'
_SAMPLE_RATE_STREAM_NAME = 'sample_rate_in'
_SAMPLE_RATE_TAG = 'SAMPLE_RATE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class AudioClassifierOptions:
  """Options for the audio classifier task.

  Attributes:
    base_options: Base options for the audio classifier task.
    running_mode: The running mode of the audio classifier. Default to the audio
      clips mode. Audio classifier has two running modes:
        1) The audio clips mode for running classification on independent audio
          clips.
        2) The audio stream mode for running classification on the audio stream,
          such as from microphone. In this mode, the "sample_rate" below must be
          provided, and the "result_callback" below must be specified to receive
          the classification results asynchronously.
    sample_rate: The sample rate of the input audios. Must be set when the
      running mode is set to RunningMode.AUDIO_STREAM.
    classifier_options: Options for the audio classification task.
    result_callback: The user-defined result callback for processing audio
      stream data. The result callback should only be specified when the running
      mode is set to RunningMode.AUDIO_STREAM.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.AUDIO_CLIPS
  sample_rate: float = -1
  classifier_options: _ClassifierOptions = _ClassifierOptions()
  result_callback: Optional[
      Callable[[classifications.ClassificationResult, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _AudioClassifierGraphOptionsProto:
    """Generates an AudioClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.AUDIO_CLIPS else True
    classifier_options_proto = self.classifier_options.to_pb2()

    return _AudioClassifierGraphOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto,
        sample_rate=self.sample_rate)


class AudioClassifier(base_audio_task_api.BaseAudioTaskApi):
  """Performs audio classification on audio clips or audio stream."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageClassifier':
    """Creates an `AudioClassifier` object from a TensorFlow Lite model and the default `AudioClassifierOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `AudioClassifier` object that's created from the model file and the
      default `AudioClassifierOptions`.

    Raises:
      ValueError: If failed to create `AudioClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = AudioClassifierOptions(
        base_options=base_options, running_mode=_RunningMode.AUDIO_CLIPS)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: AudioClassifierOptions) -> 'AudioClassifier':
    """Creates the `AudioClassifier` object from audio classifier options.

    Args:
      options: Options for the audio classifier task.

    Returns:
      `AudioClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `AudioClassifier` object from
        `AudioClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    if options.running_mode == _RunningMode.AUDIO_STREAM and options.sample_rate < 0:
      raise ValueError(
          "The audio classifier is in audio stream mode, the sample rate must "
          "be specified in the AudioClassifierOptions.")

    def packets_callback(output_packets: Mapping[str, packet.Packet]):
      classification_result_proto = classifications_pb2.ClassificationResult()
      classification_result_proto.CopyFrom(
          packet_getter.get_proto(
              output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME]))

      classification_result = classifications.ClassificationResult([
          classifications.Classifications.create_from_pb2(classification)
          for classification in classification_result_proto.classifications
      ])
      timestamp = output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME].timestamp
      options.result_callback(classification_result,
                              timestamp.value // _MICRO_SECONDS_PER_MILLISECOND)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_AUDIO_TAG, _AUDIO_IN_STREAM_NAME]),
            ':'.join([_SAMPLE_RATE_TAG, _SAMPLE_RATE_STREAM_NAME])
        ],
        output_streams=[
            ':'.join([
                _CLASSIFICATION_RESULT_TAG,
                _CLASSIFICATION_RESULT_OUT_STREAM_NAME
            ])
        ],
        task_options=options)
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode ==
            _RunningMode.LIVE_STREAM), options.running_mode,
        packets_callback if options.result_callback else None)

  def classify(
      self,
      audio_clip: matrix.Matrix,
      audio_sample_rate: float,
  ) -> classifications.ClassificationResult:
    """Sends audio data (a block in a continuous audio stream) to perform audio
    classification. Only use this method when the AudioClassifier is created
    with the audio clips running mode.

    Args:
      audio_clip: The audio clip is represented as a MediaPipe Matrix that has
        the number of channels rows and the number of samples per channel
        columns. The method accepts audio clips with various length and audio
        sample rate. It's required to provide the corresponding audio sample
        rate along with the input audio clips.
      audio_sample_rate: Sample rate of the input audio.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If audio classification failed to run.
    """
    output_packets = self._process_audio_clip({
        _AUDIO_IN_STREAM_NAME: packet_creator.create_matrix(audio_clip),
        _SAMPLE_RATE_STREAM_NAME: packet_creator.create_float(audio_sample_rate)
    })

    classification_result_proto = classifications_pb2.ClassificationResult()
    classification_result_proto.CopyFrom(
        packet_getter.get_proto(
            output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME]))

    return classifications.ClassificationResult([
        classifications.Classifications.create_from_pb2(classification)
        for classification in classification_result_proto.classifications
    ])

  def classify_async(
      self,
      audio_block: matrix.Matrix,
      audio_sample_rate: float,
      timestamp_ms: int,
  ) -> None:
    """Sends audio data (a block in a continuous audio stream) to perform audio
    classification. Only use this method when the AudioClassifier is created
    with the audio stream running mode.

    The `result_callback` provides:
      - A classification result object that contains a list of classifications.
      - The input timestamp in milliseconds.

    Args:
      audio_block: The audio block is represented as a MediaPipe Matrix that has
        the number of channels rows and the number of samples per channel
        columns. The audio data will be resampled, accumulated, and framed to
        the proper size for the underlying model to consume. It's required to
        provide a timestamp (in milliseconds) to indicate the start time of the
        input audio block. The timestamps must be monotonically increasing.
      audio_sample_rate: Sample rate of the input audio.
      timestamp_ms: The timestamp of the input audio in milliseconds.

    Raises:
      ValueError: If the current input timestamp is smaller than what the audio
        classifier has already processed.
    """
    self._send_audio_stream_data({
        _AUDIO_IN_STREAM_NAME:
            packet_creator.create_matrix(audio_block).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _SAMPLE_RATE_STREAM_NAME:
            packet_creator.create_float(audio_sample_rate).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
