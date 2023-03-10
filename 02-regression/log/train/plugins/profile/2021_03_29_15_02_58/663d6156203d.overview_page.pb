�	��\I@��\I@!��\I@	|�!��@|�!��@!|�!��@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��\I@]k�SU��?1�Hm���?A�-II�?IϠ���?Y�F�ҷ?*	�/�$�^@2F
Iterator::Model��'�߯?!���;0<I@)�i�����?1u��&=@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�c]�F�?!��O[9@)�z��?1�-$�U�4@:Preprocessing2U
Iterator::Model::ParallelMapV2[��ù�?!�ƿ��1@)[��ù�?1�ƿ��1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�I�2��?!�|�C�1@)�ek}�І?1Z�h!"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��&k�C�?!U�W @)��&k�C�?1U�W @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7p�G�?!)���H@)W	�3�z?1O���-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�K��$wx?!yP��^@)�K��$wx?1yP��^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��9�ؗ?!,HMV�2@)t#,*�tb?1��:��9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 43.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�41.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9|�!��@I��n!�U@Q4����$@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]k�SU��?]k�SU��?!]k�SU��?      ��!       "	�Hm���?�Hm���?!�Hm���?*      ��!       2	�-II�?�-II�?!�-II�?:	Ϡ���?Ϡ���?!Ϡ���?B      ��!       J	�F�ҷ?�F�ҷ?!�F�ҷ?R      ��!       Z	�F�ҷ?�F�ҷ?!�F�ҷ?b      ��!       JGPUY|�!��@b q��n!�U@y4����$@�"6
regression/dense2/MatMulMatMul�T�~>�?!�T�~>�?0"D
&gradient_tape/regression/dense2/MatMulMatMulS�y�F�?!"'��⥳?0"D
(gradient_tape/regression/dense2/MatMul_1MatMulrz_�p��?![�$q�?"D
&gradient_tape/regression/dense1/MatMulMatMul}�<�Cs�?!�Z~^��?0"6
regression/dense1/MatMulMatMul�~��	�?!�������?0"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitchc.����?!MF,��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam@��AH�?!Mn:f�.�?"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdamB���@s�?![�:?���?"Q
0gradient_tape/regression/out/BiasAdd/BiasAddGradBiasAddGradB���@s�?!��WK��?"=
 regression/normalization/truedivRealDivB���@s�?!�uW�|�?Q      Y@Y     �6@a     `S@q�W,�$
K@y�n�
R�?"�
both�Your program is POTENTIALLY input-bound because 43.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�41.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�54.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 