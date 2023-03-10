?	??O?m?	@??O?m?	@!??O?m?	@	j??h??"@j??h??"@!j??h??"@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??O?m?	@(??9??1??\???Ap?n?ꐫ?IF?~?;??Y??????*	X9??xs@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA?)V???!`??Y?P@)m?s?p??1-ćΛ?O@:Preprocessing2F
Iterator::Model?o	?????!???TK?0@)?E|'f???1?????$@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?)?D/???!<4q;?$@)?Y?H?s??1O.?e? @:Preprocessing2U
Iterator::Model::ParallelMapV2?T1?ϓ?!?L??@)?T1?ϓ?1?L??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapK?46??!|?N???Q@)? ??=@??1|#:v'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???)??!1?%?q
@)???)??11?%?q
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???I{?!?<?U@)???I{?1?<?U@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ??Û??!???*??T@)??-</{?1rj?w? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?33.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t44.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9j??h??"@Ia:>??S@Q??0?&@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(??9??(??9??!(??9??      ??!       "	??\?????\???!??\???*      ??!       2	p?n?ꐫ?p?n?ꐫ?!p?n?ꐫ?:	F?~?;??F?~?;??!F?~?;??B      ??!       J	????????????!??????R      ??!       Z	????????????!??????b      ??!       JGPUYj??h??"@b qa:>??S@y??0?&@?"6
regression/dense2/MatMulMatMul??%@???!??%@???0"D
&gradient_tape/regression/dense2/MatMulMatMul??ѽ???! o ????0"D
&gradient_tape/regression/dense1/MatMulMatMul?i5???!$:5?ռ?0"D
(gradient_tape/regression/dense2/MatMul_1MatMul??D?p??!??+3????"6
regression/dense1/MatMulMatMul??Ly5??!.QX?#??0"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitchZ?9z?e??!ُ?5? ??"Q
0gradient_tape/regression/out/BiasAdd/BiasAddGradBiasAddGrad̓?=6???!R]?????"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam??[X`??!"~_e??"=
 regression/normalization/truedivRealDiv_?m?n??!??=T???"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam????U??!8X	?????Q      Y@Y     ?6@a     `S@q?????8@y@X?Is??"?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?33.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t44.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?25.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 