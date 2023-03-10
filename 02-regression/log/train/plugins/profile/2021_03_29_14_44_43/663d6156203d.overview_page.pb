?	???Tl@???Tl@!???Tl@	?5md??@?5md??@!?5md??@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???Tl@M???????1!撪?&??AA??4F???IK?b???Y?CV???*	!?rh?=c@2F
Iterator::Model?68?ڲ?!4G???G@)?f?R@ڧ?1/???'D>@:Preprocessing2U
Iterator::Model::ParallelMapV2N4?s???!Ӆ?ë?1@)N4?s???1Ӆ?ë?1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatUMu??!??2Iu4@)V}??b??1l???C-0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Τ??!?(9p?v;@)?s?v?4??1?.?9x?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM?*?????!cxP???"@)M?*?????1cxP???"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateXr????!?"}??,@)? {?|?1?TY?`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????z?!v,U? @)?????z?1v,U? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[?? ????!?˸J@)???z?1??-i??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?41.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?5md??@IĈN??rU@QU???#@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M???????M???????!M???????      ??!       "	!撪?&??!撪?&??!!撪?&??*      ??!       2	A??4F???A??4F???!A??4F???:	K?b???K?b???!K?b???B      ??!       J	?CV????CV???!?CV???R      ??!       Z	?CV????CV???!?CV???b      ??!       JGPUY?5md??@b qĈN??rU@yU???#@?"D
&gradient_tape/regression/dense2/MatMulMatMul.??p???!.??p???0"6
regression/dense2/MatMulMatMulqGJ????!?0?^y"??0"6
regression/dense1/MatMulMatMul?J?Dd??!?3?ʛԻ?0"D
(gradient_tape/regression/dense2/MatMul_1MatMul?V???!???/??"D
&gradient_tape/regression/dense1/MatMulMatMul$??U????!??`?_??0"R
/gradient_tape/mean_absolute_error/DynamicStitchDynamicStitchǚ??????!I?
?A???"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamdo?&%???!6nۃf???"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdam?\?[0??!ꌞ??6??"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam?\?[0??!?bO?????"=
 regression/normalization/truedivRealDiv?\?[0??!?8 x???Q      Y@Y     ?6@a     `S@q??ġ?G@y?p?????"?
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?41.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?46.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 