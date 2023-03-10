�	���q��@���q��@!���q��@	`S�7�@@`S�7�@@!`S�7�@@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���q��@�C�l�;�?1B�%U�M�?A82�����?I���%:��?Y^��Nw^�?*	8�A`eҤ@2U
Iterator::Model::ParallelMapV2��Dh�?!���^&�H@)��Dh�?1���^&�H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�sF����?!�L���H@)4I,)w_�?1�:|�=�G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(֩�=#�?!%�o�!�?)x�-;�?�?1�#����?:Preprocessing2F
Iterator::ModelZf��`�?!�C�I@)�X6sHj�?1����mk�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipod�C�?!U���+�H@)*T7ۃ?14�dQ�H�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�&����?!\!���'�?)�&����?1\!���'�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^c���x?!�ayC�?)^c���x?1�ayC�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���0(��?!�t���H@)�s|�8ch?1��O�N��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 32.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"�27.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t32.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9_S�7�@@I��s�N@Q��Z�@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�C�l�;�?�C�l�;�?!�C�l�;�?      ��!       "	B�%U�M�?B�%U�M�?!B�%U�M�?*      ��!       2	82�����?82�����?!82�����?:	���%:��?���%:��?!���%:��?B      ��!       J	^��Nw^�?^��Nw^�?!^��Nw^�?R      ��!       Z	^��Nw^�?^��Nw^�?!^��Nw^�?b      ��!       JGPUY_S�7�@@b q��s�N@y��Z�@�"D
&gradient_tape/regression/dense2/MatMulMatMul�Z�N�{�?!�Z�N�{�?0"6
regression/dense2/MatMulMatMul�ΟIqѡ?!�L���?0"D
&gradient_tape/regression/dense1/MatMulMatMul�6�7G�?!�#�&J�?0"D
(gradient_tape/regression/dense2/MatMul_1MatMul�6�7G�?!4�����?"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitchG����;�?!F���E�?"6
regression/dense1/MatMulMatMulu��$�?!�������?0"Q
0gradient_tape/regression/out/BiasAdd/BiasAddGradBiasAddGrad�y��?!�}��Y+�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam��l��F�?!i�ߜ��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamA��f�	�?!��N��*�?"=
 regression/normalization/truedivRealDivA��f�	�?!�轉P��?Q      Y@Y     �6@a     `S@q�:��a6%@y�!�����?"�
host�Your program is HIGHLY input-bound because 32.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�27.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t32.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�10.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 