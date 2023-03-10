�	F��}��@F��}��@!F��}��@	K�7�D�.@K�7�D�.@!K�7�D�.@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6F��}��@w�h�h��?1*�n�EE�?A�N\�W �?Ir2q� F�?Yڏ�a�?*	��K7�q@2U
Iterator::Model::ParallelMapV2N&n�@�?!2Ɋt��G@)N&n�@�?12Ɋt��G@:Preprocessing2F
Iterator::ModelZ�Pۆ�?!,��6P@)0��\�?1N֮}�J1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�]�pX�?!�2{16@)zލ�A�?1�9"hg!@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty�0DN_�?!�����%@)n�r��?1i���� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�s]�@�?!awD�I@)�s]�@�?1awD�I@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate!��^?!9XC���*@)p$�`S�?1nO�L�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)����h}?!�Hj�C@))����h}?1�Hj�C@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zips����?!�˝L��A@)\Va3�y?1<$X�=@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 15.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�31.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t41.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9L�7�D�.@IS�N��R@Q���� �$@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	w�h�h��?w�h�h��?!w�h�h��?      ��!       "	*�n�EE�?*�n�EE�?!*�n�EE�?*      ��!       2	�N\�W �?�N\�W �?!�N\�W �?:	r2q� F�?r2q� F�?!r2q� F�?B      ��!       J	ڏ�a�?ڏ�a�?!ڏ�a�?R      ��!       Z	ڏ�a�?ڏ�a�?!ڏ�a�?b      ��!       JGPUYL�7�D�.@b qS�N��R@y���� �$@�"6
regression/dense2/MatMulMatMul�ʸ���?!�ʸ���?0"D
&gradient_tape/regression/dense2/MatMulMatMul��q��?!T�P���?0"D
(gradient_tape/regression/dense2/MatMul_1MatMul�0� )6�?!�w�֞�?"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitch�����?!���e�?"6
regression/dense1/MatMulMatMul#������?!$��MB�?0"D
&gradient_tape/regression/dense1/MatMulMatMul����F��?!v�^���?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam1��=i�?!�[q��@�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamP!6�?!�9��?"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdamP!6�?!(�z,Â�?"=
 regression/normalization/truedivRealDivP!6�?!=�����?Q      Y@Y     �6@a     `S@q�����8@y�<b(�?"�
both�Your program is MODERATELY input-bound because 15.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�31.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t41.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�24.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 