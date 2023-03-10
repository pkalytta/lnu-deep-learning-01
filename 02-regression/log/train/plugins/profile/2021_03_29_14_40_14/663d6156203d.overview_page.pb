?	m?i?*l@m?i?*l@!m?i?*l@	??8?/@??8?/@!??8?/@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6m?i?*l@FCƣT???1?~k'J??A?o?^}<??IU?	g?V??Y)@̘??*	P??n?`@2F
Iterator::ModelݲC?Ö??!?[????J@)?9τ&??1????|B@:Preprocessing2U
Iterator::Model::ParallelMapV2X S??!?J??J1@)X S??1?J??J1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatG仔?d??!P??Ni4@)>?^?????1&]??S0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceJF?v??!?.? 
?!@)JF?v??1?.? 
?!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????O???!Pp???G6@)zR&5???1-????A!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatevl?u???!r 3̏M+@)F%u?{?1????o@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor$`tys?v?!?\M?LU@)$`tys?v?1?\M?LU@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?(?	0??!$?V|FG@)??$?ptu?15????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?32.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t42.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??8?/@I&{?,??R@Q???? @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	FCƣT???FCƣT???!FCƣT???      ??!       "	?~k'J???~k'J??!?~k'J??*      ??!       2	?o?^}<???o?^}<??!?o?^}<??:	U?	g?V??U?	g?V??!U?	g?V??B      ??!       J	)@̘??)@̘??!)@̘??R      ??!       Z	)@̘??)@̘??!)@̘??b      ??!       JGPUY??8?/@b q&{?,??R@y???? @?"6
regression/dense2/MatMulMatMul?4?C+??!?4?C+??0"D
&gradient_tape/regression/dense2/MatMulMatMul???E??!?A?d/???0"D
(gradient_tape/regression/dense2/MatMul_1MatMul}@???2??!?C%?Q??"6
regression/dense1/MatMulMatMul???;???!<h?`a??0"D
&gradient_tape/regression/dense1/MatMulMatMuly$?????!????Å??0"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitch?1?K?W??!?Ҁƾ???"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam?]?X]??!?t?i'??"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam?{I??!h?????"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam?{I??!??I^???"Q
0gradient_tape/regression/out/BiasAdd/BiasAddGradBiasAddGrad?{I??!??+?2f??Q      Y@Y     ?6@a     `S@qC3ؚ??5@yr?@s???"?
both?Your program is MODERATELY input-bound because 15.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?32.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t42.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?21.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 