	��z�@��z�@!��z�@	�q?b4@�q?b4@!�q?b4@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��z�@u�^�?1���(�?A7¢"N'�?IYQ�i��?Yit�3��?*	�S㥛�u@2F
Iterator::Modelo�ŏ1�?!�Z���:J@)��
/��?1��q�)�D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�G��[�?!:z\+�m>@)�%Tp�?1l��X<�;@:Preprocessing2U
Iterator::Model::ParallelMapV2���ۂ��?!G,�T%@)���ۂ��?1G,�T%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�F�ҿ$�?!a��<�'@)��[�t�?1|����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^��I�Ԑ?!Ed'��@)^��I�Ԑ?1Ed'��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM��f�׽?!)����@@)^�c@�z�?1�Pٓ��
@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�,��;��?!kvF�V@)�,��;��?1kvF�V@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipz�I|��?!3�ipA�G@)m�i�*�y?1G~ʦ�b�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 20.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"�29.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t38.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�q?b4@I>��'EeQ@Q�֝D�$@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	u�^�?u�^�?!u�^�?      ��!       "	���(�?���(�?!���(�?*      ��!       2	7¢"N'�?7¢"N'�?!7¢"N'�?:	YQ�i��?YQ�i��?!YQ�i��?B      ��!       J	it�3��?it�3��?!it�3��?R      ��!       Z	it�3��?it�3��?!it�3��?b      ��!       JGPUY�q?b4@b q>��'EeQ@y�֝D�$@