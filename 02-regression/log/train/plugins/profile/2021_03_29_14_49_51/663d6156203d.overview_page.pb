?	?qS@?qS@!?qS@	@?B??@@?B??@!@?B??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?qS@?y ??:??1qN`:???A?C p???I kծ	??YԘsI???*	X9???m@2F
Iterator::ModelmXSYv??!??E`҅Q@)B??v?$??1??y??M@:Preprocessing2U
Iterator::Model::ParallelMapV2XY?????!l?yu?%@)XY?????1l?yu?%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??};???!,C\ݓV)@)K??z2???1??U??h$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]??ʾ+??!Vj$?-@)?Ws?`???1/?C7??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceFzQ?_??!pR??ۜ@)FzQ?_??1pR??ۜ@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?hUM??!?B???2!@)GW??:??1f?,?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?[[%x?!4v???@)?[[%x?14v???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(??P??!??~??=@)Zd;?O?w?1b?&x?:@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?38.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t42.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9@?B??@I?[?|??T@Q???D?$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?y ??:???y ??:??!?y ??:??      ??!       "	qN`:???qN`:???!qN`:???*      ??!       2	?C p????C p???!?C p???:	 kծ	?? kծ	??! kծ	??B      ??!       J	ԘsI???ԘsI???!ԘsI???R      ??!       Z	ԘsI???ԘsI???!ԘsI???b      ??!       JGPUY@?B??@b q?[?|??T@y???D?$@?"6
regression/dense2/MatMulMatMul?E?Fֶ??!?E?Fֶ??0"D
&gradient_tape/regression/dense2/MatMulMatMul???????!??C1gR??0"D
(gradient_tape/regression/dense2/MatMul_1MatMul?Tl<???!Sg3??"D
&gradient_tape/regression/dense1/MatMulMatMul?;?R???!ȇxc????0"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitch??8<.???!????? ??"6
regression/dense1/MatMulMatMul(?,???!|i?}???0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam???[??!߇?UF??"Q
0gradient_tape/regression/out/BiasAdd/BiasAddGradBiasAddGrad??w?ά??!h??????"=
 regression/normalization/truedivRealDiv?? n???!cvk ???"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam?B5????!??NQ?r??Q      Y@Y     ?6@a     `S@qbd????B@y????F??"?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?38.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t42.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?38.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 