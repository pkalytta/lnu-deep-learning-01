	�����@�����@!�����@	r���Wb&@r���Wb&@!r���Wb&@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�����@7l[�٠�?1���N$�?A;n��tˮ?I���[�?Ynē����?*	�� �r�]@2F
Iterator::Model겘�|\�?!���ig�F@)�'�X�?1��k0�<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��(�N�?!*p 9@)5F�j��?1���-�B4@:Preprocessing2U
Iterator::Model::ParallelMapV2�1=a��?!�c��Ȏ0@)�1=a��?1�c��Ȏ0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapdX��G�?!c�l�9@)LS8���?1��d�v#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceђ����?!�гX�#@)ђ����?1�гX�#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�.����?!��1��.@)W�}W�{?1����$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���2w?!���H�,@)���2w?1���H�,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�P����?!ili��aK@)�Z}uU�v?1-�Z�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 11.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�31.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t46.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9s���Wb&@Idr_-��S@Qn�V?�#@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	7l[�٠�?7l[�٠�?!7l[�٠�?      ��!       "	���N$�?���N$�?!���N$�?*      ��!       2	;n��tˮ?;n��tˮ?!;n��tˮ?:	���[�?���[�?!���[�?B      ��!       J	nē����?nē����?!nē����?R      ��!       Z	nē����?nē����?!nē����?b      ��!       JGPUYs���Wb&@b qdr_-��S@yn�V?�#@