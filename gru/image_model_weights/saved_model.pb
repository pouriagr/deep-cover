��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.9.22v2.9.1-132-g18960c44ad38թ
�
RMSprop/rnn/gru_cell/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_nameRMSprop/rnn/gru_cell/bias/rms
�
1RMSprop/rnn/gru_cell/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/rnn/gru_cell/bias/rms*
_output_shapes
:	�*
dtype0
�
)RMSprop/rnn/gru_cell/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*:
shared_name+)RMSprop/rnn/gru_cell/recurrent_kernel/rms
�
=RMSprop/rnn/gru_cell/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp)RMSprop/rnn/gru_cell/recurrent_kernel/rms*
_output_shapes
:	@�*
dtype0
�
RMSprop/rnn/gru_cell/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!RMSprop/rnn/gru_cell/kernel/rms
�
3RMSprop/rnn/gru_cell/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/rnn/gru_cell/kernel/rms*
_output_shapes
:	�*
dtype0
�
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:
*
dtype0
�
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*)
shared_nameRMSprop/dense/kernel/rms
�
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

:@
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	

rnn/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namernn/gru_cell/bias
x
%rnn/gru_cell/bias/Read/ReadVariableOpReadVariableOprnn/gru_cell/bias*
_output_shapes
:	�*
dtype0
�
rnn/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*.
shared_namernn/gru_cell/recurrent_kernel
�
1rnn/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOprnn/gru_cell/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
rnn/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_namernn/gru_cell/kernel
|
'rnn/gru_cell/kernel/Read/ReadVariableOpReadVariableOprnn/gru_cell/kernel*
_output_shapes
:	�*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@
*
dtype0

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�+
value�+B�+ B�+
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
'
$0
%1
&2
3
4*
'
$0
%1
&2
3
4*
* 
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
,trace_0
-trace_1
.trace_2
/trace_3* 
6
0trace_0
1trace_1
2trace_2
3trace_3* 
* 
z
4iter
	5decay
6learning_rate
7momentum
8rho	rmsq	rmsr	$rmss	%rmst	&rmsu*

9serving_default* 

$0
%1
&2*

$0
%1
&2*
* 
�

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
@trace_0
Atrace_1
Btrace_2
Ctrace_3* 
6
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_3* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator

$kernel
%recurrent_kernel
&bias*
* 

0
1*

0
1*
* 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

[trace_0* 

\trace_0* 
SM
VARIABLE_VALUErnn/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUErnn/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUErnn/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

]0
^1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

$0
%1
&2*

$0
%1
&2*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
h	variables
i	keras_api
	jtotal
	kcount*
H
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

j0
k1*

h	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

l	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
��
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUERMSprop/rnn/gru_cell/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE)RMSprop/rnn/gru_cell/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUERMSprop/rnn/gru_cell/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_model_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_model_inputrnn/gru_cell/biasrnn/gru_cell/kernelrnn/gru_cell/recurrent_kerneldense/kernel
dense/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_64799
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp'rnn/gru_cell/kernel/Read/ReadVariableOp1rnn/gru_cell/recurrent_kernel/Read/ReadVariableOp%rnn/gru_cell/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp3RMSprop/rnn/gru_cell/kernel/rms/Read/ReadVariableOp=RMSprop/rnn/gru_cell/recurrent_kernel/rms/Read/ReadVariableOp1RMSprop/rnn/gru_cell/bias/rms/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_66029
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasrnn/gru_cell/kernelrnn/gru_cell/recurrent_kernelrnn/gru_cell/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototal_1count_1totalcountRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/rnn/gru_cell/kernel/rms)RMSprop/rnn/gru_cell/recurrent_kernel/rmsRMSprop/rnn/gru_cell/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_66096��
�<
�
while_body_65262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�<
�
while_body_64353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�

�
@__inference_dense_layer_call_and_return_conditional_losses_64461

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�<
�
while_body_64582
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_64442

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_64353*
condR
while_cond_64352*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_64742
model_input
unknown:	�
	unknown_0:	�
	unknown_1:	@�
	unknown_2:@

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_64714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�
�
C__inference_gru_cell_layer_call_and_return_conditional_losses_64012

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������@b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������@]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������@Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������@S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������@X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
�
while_cond_64581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_64581___redundant_placeholder03
/while_while_cond_64581___redundant_placeholder13
/while_while_cond_64581___redundant_placeholder23
/while_while_cond_64581___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
#__inference_rnn_layer_call_fn_65164
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�M
�
!__inference__traced_restore_66096
file_prefix/
assignvariableop_dense_kernel:@
+
assignvariableop_1_dense_bias:
9
&assignvariableop_2_rnn_gru_cell_kernel:	�C
0assignvariableop_3_rnn_gru_cell_recurrent_kernel:	@�7
$assignvariableop_4_rnn_gru_cell_bias:	�)
assignvariableop_5_rmsprop_iter:	 *
 assignvariableop_6_rmsprop_decay: 2
(assignvariableop_7_rmsprop_learning_rate: -
#assignvariableop_8_rmsprop_momentum: (
assignvariableop_9_rmsprop_rho: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: >
,assignvariableop_14_rmsprop_dense_kernel_rms:@
8
*assignvariableop_15_rmsprop_dense_bias_rms:
F
3assignvariableop_16_rmsprop_rnn_gru_cell_kernel_rms:	�P
=assignvariableop_17_rmsprop_rnn_gru_cell_recurrent_kernel_rms:	@�D
1assignvariableop_18_rmsprop_rnn_gru_cell_bias_rms:	�
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_rnn_gru_cell_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_rnn_gru_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_rnn_gru_cell_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_rmsprop_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_rmsprop_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_rmsprop_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_rmsprop_momentumIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_rmsprop_rhoIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_rmsprop_dense_kernel_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_rmsprop_dense_bias_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_rmsprop_rnn_gru_cell_kernel_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp=assignvariableop_17_rmsprop_rnn_gru_cell_recurrent_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_rmsprop_rnn_gru_cell_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_65659

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_65570*
condR
while_cond_65569*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_64829

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	@�
	unknown_2:@

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_64714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_rnn_layer_call_fn_65186

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_64352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_64352___redundant_placeholder03
/while_while_cond_64352___redundant_placeholder13
/while_while_cond_64352___redundant_placeholder23
/while_while_cond_64352___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�3
�
>__inference_rnn_layer_call_and_return_conditional_losses_64273

inputs!
gru_cell_64196:	�!
gru_cell_64198:	�!
gru_cell_64200:	@�
identity�� gru_cell/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_64196gru_cell_64198gru_cell_64200*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64156n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_64196gru_cell_64198gru_cell_64200*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_64209*
condR
while_cond_64208*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@q
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
C__inference_gru_cell_layer_call_and_return_conditional_losses_64156

inputs

states*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������@b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������@]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������@Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������@S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������@X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�.
�
__inference__traced_save_66029
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop2
.savev2_rnn_gru_cell_kernel_read_readvariableop<
8savev2_rnn_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_rnn_gru_cell_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop>
:savev2_rmsprop_rnn_gru_cell_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_rnn_gru_cell_recurrent_kernel_rms_read_readvariableop<
8savev2_rmsprop_rnn_gru_cell_bias_rms_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop.savev2_rnn_gru_cell_kernel_read_readvariableop8savev2_rnn_gru_cell_recurrent_kernel_read_readvariableop,savev2_rnn_gru_cell_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop:savev2_rmsprop_rnn_gru_cell_kernel_rms_read_readvariableopDsavev2_rmsprop_rnn_gru_cell_recurrent_kernel_rms_read_readvariableop8savev2_rmsprop_rnn_gru_cell_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapesz
x: :@
:
:	�:	@�:	�: : : : : : : : : :@
:
:	�:	@�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@
: 

_output_shapes
:
:%!

_output_shapes
:	�:%!

_output_shapes
:	@�:%!

_output_shapes
:	�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@
: 

_output_shapes
:
:%!

_output_shapes
:	�:%!

_output_shapes
:	@�:%!

_output_shapes
:	�:

_output_shapes
: 
�<
�
while_body_65570
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�A
�
rnn_while_body_64894$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0?
,rnn_while_gru_cell_readvariableop_resource_0:	�F
3rnn_while_gru_cell_matmul_readvariableop_resource_0:	�H
5rnn_while_gru_cell_matmul_1_readvariableop_resource_0:	@�
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor=
*rnn_while_gru_cell_readvariableop_resource:	�D
1rnn_while_gru_cell_matmul_readvariableop_resource:	�F
3rnn_while_gru_cell_matmul_1_readvariableop_resource:	@���(rnn/while/gru_cell/MatMul/ReadVariableOp�*rnn/while/gru_cell/MatMul_1/ReadVariableOp�!rnn/while/gru_cell/ReadVariableOp�
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
!rnn/while/gru_cell/ReadVariableOpReadVariableOp,rnn_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
rnn/while/gru_cell/unstackUnpack)rnn/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(rnn/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3rnn_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
rnn/while/gru_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:00rnn/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/while/gru_cell/BiasAddBiasAdd#rnn/while/gru_cell/MatMul:product:0#rnn/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������m
"rnn/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/while/gru_cell/splitSplit+rnn/while/gru_cell/split/split_dim:output:0#rnn/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
*rnn/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5rnn_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
rnn/while/gru_cell/MatMul_1MatMulrnn_while_placeholder_22rnn/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/while/gru_cell/BiasAdd_1BiasAdd%rnn/while/gru_cell/MatMul_1:product:0#rnn/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������m
rnn/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����o
$rnn/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/while/gru_cell/split_1SplitV%rnn/while/gru_cell/BiasAdd_1:output:0!rnn/while/gru_cell/Const:output:0-rnn/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
rnn/while/gru_cell/addAddV2!rnn/while/gru_cell/split:output:0#rnn/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@s
rnn/while/gru_cell/SigmoidSigmoidrnn/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_1AddV2!rnn/while/gru_cell/split:output:1#rnn/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@w
rnn/while/gru_cell/Sigmoid_1Sigmoidrnn/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mulMul rnn/while/gru_cell/Sigmoid_1:y:0#rnn/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_2AddV2!rnn/while/gru_cell/split:output:2rnn/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@o
rnn/while/gru_cell/TanhTanhrnn/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mul_1Mulrnn/while/gru_cell/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:���������@]
rnn/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
rnn/while/gru_cell/subSub!rnn/while/gru_cell/sub/x:output:0rnn/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mul_2Mulrnn/while/gru_cell/sub:z:0rnn/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_3AddV2rnn/while/gru_cell/mul_1:z:0rnn/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@v
4rnn/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1=rnn/while/TensorArrayV2Write/TensorListSetItem/index:output:0rnn/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: �
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: �
rnn/while/Identity_4Identityrnn/while/gru_cell/add_3:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:���������@�
rnn/while/NoOpNoOp)^rnn/while/gru_cell/MatMul/ReadVariableOp+^rnn/while/gru_cell/MatMul_1/ReadVariableOp"^rnn/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3rnn_while_gru_cell_matmul_1_readvariableop_resource5rnn_while_gru_cell_matmul_1_readvariableop_resource_0"h
1rnn_while_gru_cell_matmul_readvariableop_resource3rnn_while_gru_cell_matmul_readvariableop_resource_0"Z
*rnn_while_gru_cell_readvariableop_resource,rnn_while_gru_cell_readvariableop_resource_0"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"�
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2T
(rnn/while/gru_cell/MatMul/ReadVariableOp(rnn/while/gru_cell/MatMul/ReadVariableOp2X
*rnn/while/gru_cell/MatMul_1/ReadVariableOp*rnn/while/gru_cell/MatMul_1/ReadVariableOp2F
!rnn/while/gru_cell/ReadVariableOp!rnn/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�	
�
model_rnn_while_cond_638440
,model_rnn_while_model_rnn_while_loop_counter6
2model_rnn_while_model_rnn_while_maximum_iterations
model_rnn_while_placeholder!
model_rnn_while_placeholder_1!
model_rnn_while_placeholder_22
.model_rnn_while_less_model_rnn_strided_slice_1G
Cmodel_rnn_while_model_rnn_while_cond_63844___redundant_placeholder0G
Cmodel_rnn_while_model_rnn_while_cond_63844___redundant_placeholder1G
Cmodel_rnn_while_model_rnn_while_cond_63844___redundant_placeholder2G
Cmodel_rnn_while_model_rnn_while_cond_63844___redundant_placeholder3
model_rnn_while_identity
�
model/rnn/while/LessLessmodel_rnn_while_placeholder.model_rnn_while_less_model_rnn_strided_slice_1*
T0*
_output_shapes
: _
model/rnn/while/IdentityIdentitymodel/rnn/while/Less:z:0*
T0
*
_output_shapes
: "=
model_rnn_while_identity!model/rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_64208
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_64208___redundant_placeholder03
/while_while_cond_64208___redundant_placeholder13
/while_while_cond_64208___redundant_placeholder23
/while_while_cond_64208___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
@__inference_model_layer_call_and_return_conditional_losses_64776
model_input
	rnn_64762:	�
	rnn_64764:	�
	rnn_64766:	@�
dense_64769:@

dense_64771:

identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallmodel_input	rnn_64762	rnn_64764	rnn_64766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64671�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_64769dense_64771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_64461�
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_64472o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�Y
�
@__inference_model_layer_call_and_return_conditional_losses_64991

inputs7
$rnn_gru_cell_readvariableop_resource:	�>
+rnn_gru_cell_matmul_readvariableop_resource:	�@
-rnn_gru_cell_matmul_1_readvariableop_resource:	@�6
$dense_matmul_readvariableop_resource:@
3
%dense_biasadd_readvariableop_resource:

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�"rnn/gru_cell/MatMul/ReadVariableOp�$rnn/gru_cell/MatMul_1/ReadVariableOp�rnn/gru_cell/ReadVariableOp�	rnn/while?
	rnn/ShapeShapeinputs*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:���������@g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
rnn/transpose	Transposeinputsrnn/transpose/perm:output:0*
T0*+
_output_shapes
:���������L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
rnn/gru_cell/ReadVariableOpReadVariableOp$rnn_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0{
rnn/gru_cell/unstackUnpack#rnn/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
"rnn/gru_cell/MatMul/ReadVariableOpReadVariableOp+rnn_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
rnn/gru_cell/MatMulMatMulrnn/strided_slice_2:output:0*rnn/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/gru_cell/BiasAddBiasAddrnn/gru_cell/MatMul:product:0rnn/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������g
rnn/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/gru_cell/splitSplit%rnn/gru_cell/split/split_dim:output:0rnn/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
$rnn/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-rnn_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
rnn/gru_cell/MatMul_1MatMulrnn/zeros:output:0,rnn/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/gru_cell/BiasAdd_1BiasAddrnn/gru_cell/MatMul_1:product:0rnn/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������g
rnn/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����i
rnn/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/gru_cell/split_1SplitVrnn/gru_cell/BiasAdd_1:output:0rnn/gru_cell/Const:output:0'rnn/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
rnn/gru_cell/addAddV2rnn/gru_cell/split:output:0rnn/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@g
rnn/gru_cell/SigmoidSigmoidrnn/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/add_1AddV2rnn/gru_cell/split:output:1rnn/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@k
rnn/gru_cell/Sigmoid_1Sigmoidrnn/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/mulMulrnn/gru_cell/Sigmoid_1:y:0rnn/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/add_2AddV2rnn/gru_cell/split:output:2rnn/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@c
rnn/gru_cell/TanhTanhrnn/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@y
rnn/gru_cell/mul_1Mulrnn/gru_cell/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:���������@W
rnn/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
rnn/gru_cell/subSubrnn/gru_cell/sub/x:output:0rnn/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@x
rnn/gru_cell/mul_2Mulrnn/gru_cell/sub:z:0rnn/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@}
rnn/gru_cell/add_3AddV2rnn/gru_cell/mul_1:z:0rnn/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   b
 rnn/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0)rnn/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0$rnn_gru_cell_readvariableop_resource+rnn_gru_cell_matmul_readvariableop_resource-rnn_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( * 
bodyR
rnn_while_body_64894* 
condR
rnn_while_cond_64893*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsl
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0�
dense/MatMulMatMulrnn/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
f
softmax/SoftmaxSoftmaxdense/Relu:activations:0*
T0*'
_output_shapes
:���������
h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^rnn/gru_cell/MatMul/ReadVariableOp%^rnn/gru_cell/MatMul_1/ReadVariableOp^rnn/gru_cell/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"rnn/gru_cell/MatMul/ReadVariableOp"rnn/gru_cell/MatMul/ReadVariableOp2L
$rnn/gru_cell/MatMul_1/ReadVariableOp$rnn/gru_cell/MatMul_1/ReadVariableOp2:
rnn/gru_cell/ReadVariableOprnn/gru_cell/ReadVariableOp2
	rnn/while	rnn/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�b
�
 __inference__wrapped_model_63942
model_input=
*model_rnn_gru_cell_readvariableop_resource:	�D
1model_rnn_gru_cell_matmul_readvariableop_resource:	�F
3model_rnn_gru_cell_matmul_1_readvariableop_resource:	@�<
*model_dense_matmul_readvariableop_resource:@
9
+model_dense_biasadd_readvariableop_resource:

identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�(model/rnn/gru_cell/MatMul/ReadVariableOp�*model/rnn/gru_cell/MatMul_1/ReadVariableOp�!model/rnn/gru_cell/ReadVariableOp�model/rnn/whileJ
model/rnn/ShapeShapemodel_input*
T0*
_output_shapes
:g
model/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
model/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
model/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/rnn/strided_sliceStridedSlicemodel/rnn/Shape:output:0&model/rnn/strided_slice/stack:output:0(model/rnn/strided_slice/stack_1:output:0(model/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
model/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@�
model/rnn/zeros/packedPack model/rnn/strided_slice:output:0!model/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Z
model/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model/rnn/zerosFillmodel/rnn/zeros/packed:output:0model/rnn/zeros/Const:output:0*
T0*'
_output_shapes
:���������@m
model/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/rnn/transpose	Transposemodel_input!model/rnn/transpose/perm:output:0*
T0*+
_output_shapes
:���������X
model/rnn/Shape_1Shapemodel/rnn/transpose:y:0*
T0*
_output_shapes
:i
model/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/rnn/strided_slice_1StridedSlicemodel/rnn/Shape_1:output:0(model/rnn/strided_slice_1/stack:output:0*model/rnn/strided_slice_1/stack_1:output:0*model/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
%model/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/rnn/TensorArrayV2TensorListReserve.model/rnn/TensorArrayV2/element_shape:output:0"model/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
?model/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
1model/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/rnn/transpose:y:0Hmodel/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
model/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!model/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!model/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/rnn/strided_slice_2StridedSlicemodel/rnn/transpose:y:0(model/rnn/strided_slice_2/stack:output:0*model/rnn/strided_slice_2/stack_1:output:0*model/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!model/rnn/gru_cell/ReadVariableOpReadVariableOp*model_rnn_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/rnn/gru_cell/unstackUnpack)model/rnn/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(model/rnn/gru_cell/MatMul/ReadVariableOpReadVariableOp1model_rnn_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/rnn/gru_cell/MatMulMatMul"model/rnn/strided_slice_2:output:00model/rnn/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/rnn/gru_cell/BiasAddBiasAdd#model/rnn/gru_cell/MatMul:product:0#model/rnn/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������m
"model/rnn/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/rnn/gru_cell/splitSplit+model/rnn/gru_cell/split/split_dim:output:0#model/rnn/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
*model/rnn/gru_cell/MatMul_1/ReadVariableOpReadVariableOp3model_rnn_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
model/rnn/gru_cell/MatMul_1MatMulmodel/rnn/zeros:output:02model/rnn/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/rnn/gru_cell/BiasAdd_1BiasAdd%model/rnn/gru_cell/MatMul_1:product:0#model/rnn/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������m
model/rnn/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����o
$model/rnn/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/rnn/gru_cell/split_1SplitV%model/rnn/gru_cell/BiasAdd_1:output:0!model/rnn/gru_cell/Const:output:0-model/rnn/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
model/rnn/gru_cell/addAddV2!model/rnn/gru_cell/split:output:0#model/rnn/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@s
model/rnn/gru_cell/SigmoidSigmoidmodel/rnn/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/add_1AddV2!model/rnn/gru_cell/split:output:1#model/rnn/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@w
model/rnn/gru_cell/Sigmoid_1Sigmoidmodel/rnn/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/mulMul model/rnn/gru_cell/Sigmoid_1:y:0#model/rnn/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/add_2AddV2!model/rnn/gru_cell/split:output:2model/rnn/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@o
model/rnn/gru_cell/TanhTanhmodel/rnn/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/mul_1Mulmodel/rnn/gru_cell/Sigmoid:y:0model/rnn/zeros:output:0*
T0*'
_output_shapes
:���������@]
model/rnn/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/rnn/gru_cell/subSub!model/rnn/gru_cell/sub/x:output:0model/rnn/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/mul_2Mulmodel/rnn/gru_cell/sub:z:0model/rnn/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
model/rnn/gru_cell/add_3AddV2model/rnn/gru_cell/mul_1:z:0model/rnn/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@x
'model/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   h
&model/rnn/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
model/rnn/TensorArrayV2_1TensorListReserve0model/rnn/TensorArrayV2_1/element_shape:output:0/model/rnn/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���P
model/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : m
"model/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
model/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
model/rnn/whileWhile%model/rnn/while/loop_counter:output:0+model/rnn/while/maximum_iterations:output:0model/rnn/time:output:0"model/rnn/TensorArrayV2_1:handle:0model/rnn/zeros:output:0"model/rnn/strided_slice_1:output:0Amodel/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0*model_rnn_gru_cell_readvariableop_resource1model_rnn_gru_cell_matmul_readvariableop_resource3model_rnn_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *&
bodyR
model_rnn_while_body_63845*&
condR
model_rnn_while_cond_63844*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
:model/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
,model/rnn/TensorArrayV2Stack/TensorListStackTensorListStackmodel/rnn/while:output:3Cmodel/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsr
model/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������k
!model/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!model/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/rnn/strided_slice_3StridedSlice5model/rnn/TensorArrayV2Stack/TensorListStack:tensor:0(model/rnn/strided_slice_3/stack:output:0*model/rnn/strided_slice_3/stack_1:output:0*model/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_masko
model/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
model/rnn/transpose_1	Transpose5model/rnn/TensorArrayV2Stack/TensorListStack:tensor:0#model/rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0�
model/dense/MatMulMatMul"model/rnn/strided_slice_3:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
r
model/softmax/SoftmaxSoftmaxmodel/dense/Relu:activations:0*
T0*'
_output_shapes
:���������
n
IdentityIdentitymodel/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp)^model/rnn/gru_cell/MatMul/ReadVariableOp+^model/rnn/gru_cell/MatMul_1/ReadVariableOp"^model/rnn/gru_cell/ReadVariableOp^model/rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2T
(model/rnn/gru_cell/MatMul/ReadVariableOp(model/rnn/gru_cell/MatMul/ReadVariableOp2X
*model/rnn/gru_cell/MatMul_1/ReadVariableOp*model/rnn/gru_cell/MatMul_1/ReadVariableOp2F
!model/rnn/gru_cell/ReadVariableOp!model/rnn/gru_cell/ReadVariableOp2"
model/rnn/whilemodel/rnn/while:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�
�
@__inference_model_layer_call_and_return_conditional_losses_64759
model_input
	rnn_64745:	�
	rnn_64747:	�
	rnn_64749:	@�
dense_64752:@

dense_64754:

identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallmodel_input	rnn_64745	rnn_64747	rnn_64749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64442�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_64752dense_64754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_64461�
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_64472o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�<
�
while_body_65416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�Y
�
@__inference_model_layer_call_and_return_conditional_losses_65153

inputs7
$rnn_gru_cell_readvariableop_resource:	�>
+rnn_gru_cell_matmul_readvariableop_resource:	�@
-rnn_gru_cell_matmul_1_readvariableop_resource:	@�6
$dense_matmul_readvariableop_resource:@
3
%dense_biasadd_readvariableop_resource:

identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�"rnn/gru_cell/MatMul/ReadVariableOp�$rnn/gru_cell/MatMul_1/ReadVariableOp�rnn/gru_cell/ReadVariableOp�	rnn/while?
	rnn/ShapeShapeinputs*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    x
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*'
_output_shapes
:���������@g
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
rnn/transpose	Transposeinputsrnn/transpose/perm:output:0*
T0*+
_output_shapes
:���������L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���c
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
rnn/gru_cell/ReadVariableOpReadVariableOp$rnn_gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0{
rnn/gru_cell/unstackUnpack#rnn/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
"rnn/gru_cell/MatMul/ReadVariableOpReadVariableOp+rnn_gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
rnn/gru_cell/MatMulMatMulrnn/strided_slice_2:output:0*rnn/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/gru_cell/BiasAddBiasAddrnn/gru_cell/MatMul:product:0rnn/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������g
rnn/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/gru_cell/splitSplit%rnn/gru_cell/split/split_dim:output:0rnn/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
$rnn/gru_cell/MatMul_1/ReadVariableOpReadVariableOp-rnn_gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
rnn/gru_cell/MatMul_1MatMulrnn/zeros:output:0,rnn/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/gru_cell/BiasAdd_1BiasAddrnn/gru_cell/MatMul_1:product:0rnn/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������g
rnn/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����i
rnn/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/gru_cell/split_1SplitVrnn/gru_cell/BiasAdd_1:output:0rnn/gru_cell/Const:output:0'rnn/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
rnn/gru_cell/addAddV2rnn/gru_cell/split:output:0rnn/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@g
rnn/gru_cell/SigmoidSigmoidrnn/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/add_1AddV2rnn/gru_cell/split:output:1rnn/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@k
rnn/gru_cell/Sigmoid_1Sigmoidrnn/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/mulMulrnn/gru_cell/Sigmoid_1:y:0rnn/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
rnn/gru_cell/add_2AddV2rnn/gru_cell/split:output:2rnn/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@c
rnn/gru_cell/TanhTanhrnn/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@y
rnn/gru_cell/mul_1Mulrnn/gru_cell/Sigmoid:y:0rnn/zeros:output:0*
T0*'
_output_shapes
:���������@W
rnn/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
rnn/gru_cell/subSubrnn/gru_cell/sub/x:output:0rnn/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@x
rnn/gru_cell/mul_2Mulrnn/gru_cell/sub:z:0rnn/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@}
rnn/gru_cell/add_3AddV2rnn/gru_cell/mul_1:z:0rnn/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   b
 rnn/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0)rnn/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���J
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������X
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0$rnn_gru_cell_readvariableop_resource+rnn_gru_cell_matmul_readvariableop_resource-rnn_gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( * 
bodyR
rnn_while_body_65056* 
condR
rnn_while_cond_65055*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsl
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������e
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0�
dense/MatMulMatMulrnn/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
f
softmax/SoftmaxSoftmaxdense/Relu:activations:0*
T0*'
_output_shapes
:���������
h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp#^rnn/gru_cell/MatMul/ReadVariableOp%^rnn/gru_cell/MatMul_1/ReadVariableOp^rnn/gru_cell/ReadVariableOp
^rnn/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2H
"rnn/gru_cell/MatMul/ReadVariableOp"rnn/gru_cell/MatMul/ReadVariableOp2L
$rnn/gru_cell/MatMul_1/ReadVariableOp$rnn/gru_cell/MatMul_1/ReadVariableOp2:
rnn/gru_cell/ReadVariableOprnn/gru_cell/ReadVariableOp2
	rnn/while	rnn/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
C
'__inference_softmax_layer_call_fn_65838

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_64472`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
rnn_while_cond_65055$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_65055___redundant_placeholder0;
7rnn_while_rnn_while_cond_65055___redundant_placeholder1;
7rnn_while_rnn_while_cond_65055___redundant_placeholder2;
7rnn_while_rnn_while_cond_65055___redundant_placeholder3
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_65261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_65261___redundant_placeholder03
/while_while_cond_65261___redundant_placeholder13
/while_while_cond_65261___redundant_placeholder23
/while_while_cond_65261___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_65505
inputs_03
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_65416*
condR
while_cond_65415*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
(__inference_gru_cell_layer_call_fn_65857

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64012o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�
�
@__inference_model_layer_call_and_return_conditional_losses_64475

inputs
	rnn_64443:	�
	rnn_64445:	�
	rnn_64447:	@�
dense_64462:@

dense_64464:

identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinputs	rnn_64443	rnn_64445	rnn_64447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64442�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_64462dense_64464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_64461�
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_64472o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_65351
inputs_03
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_65262*
condR
while_cond_65261*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�H
�

model_rnn_while_body_638450
,model_rnn_while_model_rnn_while_loop_counter6
2model_rnn_while_model_rnn_while_maximum_iterations
model_rnn_while_placeholder!
model_rnn_while_placeholder_1!
model_rnn_while_placeholder_2/
+model_rnn_while_model_rnn_strided_slice_1_0k
gmodel_rnn_while_tensorarrayv2read_tensorlistgetitem_model_rnn_tensorarrayunstack_tensorlistfromtensor_0E
2model_rnn_while_gru_cell_readvariableop_resource_0:	�L
9model_rnn_while_gru_cell_matmul_readvariableop_resource_0:	�N
;model_rnn_while_gru_cell_matmul_1_readvariableop_resource_0:	@�
model_rnn_while_identity
model_rnn_while_identity_1
model_rnn_while_identity_2
model_rnn_while_identity_3
model_rnn_while_identity_4-
)model_rnn_while_model_rnn_strided_slice_1i
emodel_rnn_while_tensorarrayv2read_tensorlistgetitem_model_rnn_tensorarrayunstack_tensorlistfromtensorC
0model_rnn_while_gru_cell_readvariableop_resource:	�J
7model_rnn_while_gru_cell_matmul_readvariableop_resource:	�L
9model_rnn_while_gru_cell_matmul_1_readvariableop_resource:	@���.model/rnn/while/gru_cell/MatMul/ReadVariableOp�0model/rnn/while/gru_cell/MatMul_1/ReadVariableOp�'model/rnn/while/gru_cell/ReadVariableOp�
Amodel/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
3model/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemgmodel_rnn_while_tensorarrayv2read_tensorlistgetitem_model_rnn_tensorarrayunstack_tensorlistfromtensor_0model_rnn_while_placeholderJmodel/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'model/rnn/while/gru_cell/ReadVariableOpReadVariableOp2model_rnn_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
 model/rnn/while/gru_cell/unstackUnpack/model/rnn/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
.model/rnn/while/gru_cell/MatMul/ReadVariableOpReadVariableOp9model_rnn_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
model/rnn/while/gru_cell/MatMulMatMul:model/rnn/while/TensorArrayV2Read/TensorListGetItem:item:06model/rnn/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 model/rnn/while/gru_cell/BiasAddBiasAdd)model/rnn/while/gru_cell/MatMul:product:0)model/rnn/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������s
(model/rnn/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
model/rnn/while/gru_cell/splitSplit1model/rnn/while/gru_cell/split/split_dim:output:0)model/rnn/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
0model/rnn/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;model_rnn_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
!model/rnn/while/gru_cell/MatMul_1MatMulmodel_rnn_while_placeholder_28model/rnn/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/rnn/while/gru_cell/BiasAdd_1BiasAdd+model/rnn/while/gru_cell/MatMul_1:product:0)model/rnn/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������s
model/rnn/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����u
*model/rnn/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
 model/rnn/while/gru_cell/split_1SplitV+model/rnn/while/gru_cell/BiasAdd_1:output:0'model/rnn/while/gru_cell/Const:output:03model/rnn/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
model/rnn/while/gru_cell/addAddV2'model/rnn/while/gru_cell/split:output:0)model/rnn/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@
 model/rnn/while/gru_cell/SigmoidSigmoid model/rnn/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/add_1AddV2'model/rnn/while/gru_cell/split:output:1)model/rnn/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@�
"model/rnn/while/gru_cell/Sigmoid_1Sigmoid"model/rnn/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/mulMul&model/rnn/while/gru_cell/Sigmoid_1:y:0)model/rnn/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/add_2AddV2'model/rnn/while/gru_cell/split:output:2 model/rnn/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@{
model/rnn/while/gru_cell/TanhTanh"model/rnn/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/mul_1Mul$model/rnn/while/gru_cell/Sigmoid:y:0model_rnn_while_placeholder_2*
T0*'
_output_shapes
:���������@c
model/rnn/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/rnn/while/gru_cell/subSub'model/rnn/while/gru_cell/sub/x:output:0$model/rnn/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/mul_2Mul model/rnn/while/gru_cell/sub:z:0!model/rnn/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
model/rnn/while/gru_cell/add_3AddV2"model/rnn/while/gru_cell/mul_1:z:0"model/rnn/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@|
:model/rnn/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
4model/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmodel_rnn_while_placeholder_1Cmodel/rnn/while/TensorArrayV2Write/TensorListSetItem/index:output:0"model/rnn/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���W
model/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :z
model/rnn/while/addAddV2model_rnn_while_placeholdermodel/rnn/while/add/y:output:0*
T0*
_output_shapes
: Y
model/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/rnn/while/add_1AddV2,model_rnn_while_model_rnn_while_loop_counter model/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: w
model/rnn/while/IdentityIdentitymodel/rnn/while/add_1:z:0^model/rnn/while/NoOp*
T0*
_output_shapes
: �
model/rnn/while/Identity_1Identity2model_rnn_while_model_rnn_while_maximum_iterations^model/rnn/while/NoOp*
T0*
_output_shapes
: w
model/rnn/while/Identity_2Identitymodel/rnn/while/add:z:0^model/rnn/while/NoOp*
T0*
_output_shapes
: �
model/rnn/while/Identity_3IdentityDmodel/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model/rnn/while/NoOp*
T0*
_output_shapes
: �
model/rnn/while/Identity_4Identity"model/rnn/while/gru_cell/add_3:z:0^model/rnn/while/NoOp*
T0*'
_output_shapes
:���������@�
model/rnn/while/NoOpNoOp/^model/rnn/while/gru_cell/MatMul/ReadVariableOp1^model/rnn/while/gru_cell/MatMul_1/ReadVariableOp(^model/rnn/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "x
9model_rnn_while_gru_cell_matmul_1_readvariableop_resource;model_rnn_while_gru_cell_matmul_1_readvariableop_resource_0"t
7model_rnn_while_gru_cell_matmul_readvariableop_resource9model_rnn_while_gru_cell_matmul_readvariableop_resource_0"f
0model_rnn_while_gru_cell_readvariableop_resource2model_rnn_while_gru_cell_readvariableop_resource_0"=
model_rnn_while_identity!model/rnn/while/Identity:output:0"A
model_rnn_while_identity_1#model/rnn/while/Identity_1:output:0"A
model_rnn_while_identity_2#model/rnn/while/Identity_2:output:0"A
model_rnn_while_identity_3#model/rnn/while/Identity_3:output:0"A
model_rnn_while_identity_4#model/rnn/while/Identity_4:output:0"X
)model_rnn_while_model_rnn_strided_slice_1+model_rnn_while_model_rnn_strided_slice_1_0"�
emodel_rnn_while_tensorarrayv2read_tensorlistgetitem_model_rnn_tensorarrayunstack_tensorlistfromtensorgmodel_rnn_while_tensorarrayv2read_tensorlistgetitem_model_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2`
.model/rnn/while/gru_cell/MatMul/ReadVariableOp.model/rnn/while/gru_cell/MatMul/ReadVariableOp2d
0model/rnn/while/gru_cell/MatMul_1/ReadVariableOp0model/rnn/while/gru_cell/MatMul_1/ReadVariableOp2R
'model/rnn/while/gru_cell/ReadVariableOp'model/rnn/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
^
B__inference_softmax_layer_call_and_return_conditional_losses_64472

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
while_body_64026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_gru_cell_64048_0:	�)
while_gru_cell_64050_0:	�)
while_gru_cell_64052_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_gru_cell_64048:	�'
while_gru_cell_64050:	�'
while_gru_cell_64052:	@���&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_64048_0while_gru_cell_64050_0while_gru_cell_64052_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64012r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ".
while_gru_cell_64048while_gru_cell_64048_0".
while_gru_cell_64050while_gru_cell_64050_0".
while_gru_cell_64052while_gru_cell_64052_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_64671

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_64582*
condR
while_cond_64581*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
rnn_while_cond_64893$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2&
"rnn_while_less_rnn_strided_slice_1;
7rnn_while_rnn_while_cond_64893___redundant_placeholder0;
7rnn_while_rnn_while_cond_64893___redundant_placeholder1;
7rnn_while_rnn_while_cond_64893___redundant_placeholder2;
7rnn_while_rnn_while_cond_64893___redundant_placeholder3
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�<
�
while_body_65724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0;
(while_gru_cell_readvariableop_resource_0:	�B
/while_gru_cell_matmul_readvariableop_resource_0:	�D
1while_gru_cell_matmul_1_readvariableop_resource_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor9
&while_gru_cell_readvariableop_resource:	�@
-while_gru_cell_matmul_readvariableop_resource:	�B
/while_gru_cell_matmul_1_readvariableop_resource:	@���$while/gru_cell/MatMul/ReadVariableOp�&while/gru_cell/MatMul_1/ReadVariableOp�while/gru_cell/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp/while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������i
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp1while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������i
while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����k
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@k
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@o
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@g
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:���������@Y
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@~
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:���������@�

while/NoOpNoOp%^while/gru_cell/MatMul/ReadVariableOp'^while/gru_cell/MatMul_1/ReadVariableOp^while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "d
/while_gru_cell_matmul_1_readvariableop_resource1while_gru_cell_matmul_1_readvariableop_resource_0"`
-while_gru_cell_matmul_readvariableop_resource/while_gru_cell_matmul_readvariableop_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2L
$while/gru_cell/MatMul/ReadVariableOp$while/gru_cell/MatMul/ReadVariableOp2P
&while/gru_cell/MatMul_1/ReadVariableOp&while/gru_cell/MatMul_1/ReadVariableOp2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�3
�
>__inference_rnn_layer_call_and_return_conditional_losses_64090

inputs!
gru_cell_64013:	�!
gru_cell_64015:	�!
gru_cell_64017:	@�
identity�� gru_cell/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_64013gru_cell_64015gru_cell_64017*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64012n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_64013gru_cell_64015gru_cell_64017*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_64026*
condR
while_cond_64025*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@q
NoOpNoOp!^gru_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
#__inference_rnn_layer_call_fn_65197

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�A
�
rnn_while_body_65056$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0?
,rnn_while_gru_cell_readvariableop_resource_0:	�F
3rnn_while_gru_cell_matmul_readvariableop_resource_0:	�H
5rnn_while_gru_cell_matmul_1_readvariableop_resource_0:	@�
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor=
*rnn_while_gru_cell_readvariableop_resource:	�D
1rnn_while_gru_cell_matmul_readvariableop_resource:	�F
3rnn_while_gru_cell_matmul_1_readvariableop_resource:	@���(rnn/while/gru_cell/MatMul/ReadVariableOp�*rnn/while/gru_cell/MatMul_1/ReadVariableOp�!rnn/while/gru_cell/ReadVariableOp�
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
!rnn/while/gru_cell/ReadVariableOpReadVariableOp,rnn_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
rnn/while/gru_cell/unstackUnpack)rnn/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
(rnn/while/gru_cell/MatMul/ReadVariableOpReadVariableOp3rnn_while_gru_cell_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
rnn/while/gru_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:00rnn/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/while/gru_cell/BiasAddBiasAdd#rnn/while/gru_cell/MatMul:product:0#rnn/while/gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������m
"rnn/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/while/gru_cell/splitSplit+rnn/while/gru_cell/split/split_dim:output:0#rnn/while/gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
*rnn/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOp5rnn_while_gru_cell_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype0�
rnn/while/gru_cell/MatMul_1MatMulrnn_while_placeholder_22rnn/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
rnn/while/gru_cell/BiasAdd_1BiasAdd%rnn/while/gru_cell/MatMul_1:product:0#rnn/while/gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������m
rnn/while/gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����o
$rnn/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rnn/while/gru_cell/split_1SplitV%rnn/while/gru_cell/BiasAdd_1:output:0!rnn/while/gru_cell/Const:output:0-rnn/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
rnn/while/gru_cell/addAddV2!rnn/while/gru_cell/split:output:0#rnn/while/gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@s
rnn/while/gru_cell/SigmoidSigmoidrnn/while/gru_cell/add:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_1AddV2!rnn/while/gru_cell/split:output:1#rnn/while/gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@w
rnn/while/gru_cell/Sigmoid_1Sigmoidrnn/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mulMul rnn/while/gru_cell/Sigmoid_1:y:0#rnn/while/gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_2AddV2!rnn/while/gru_cell/split:output:2rnn/while/gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@o
rnn/while/gru_cell/TanhTanhrnn/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mul_1Mulrnn/while/gru_cell/Sigmoid:y:0rnn_while_placeholder_2*
T0*'
_output_shapes
:���������@]
rnn/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
rnn/while/gru_cell/subSub!rnn/while/gru_cell/sub/x:output:0rnn/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/mul_2Mulrnn/while/gru_cell/sub:z:0rnn/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@�
rnn/while/gru_cell/add_3AddV2rnn/while/gru_cell/mul_1:z:0rnn/while/gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@v
4rnn/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1=rnn/while/TensorArrayV2Write/TensorListSetItem/index:output:0rnn/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype0:���Q
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: �
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: �
rnn/while/Identity_4Identityrnn/while/gru_cell/add_3:z:0^rnn/while/NoOp*
T0*'
_output_shapes
:���������@�
rnn/while/NoOpNoOp)^rnn/while/gru_cell/MatMul/ReadVariableOp+^rnn/while/gru_cell/MatMul_1/ReadVariableOp"^rnn/while/gru_cell/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "l
3rnn_while_gru_cell_matmul_1_readvariableop_resource5rnn_while_gru_cell_matmul_1_readvariableop_resource_0"h
1rnn_while_gru_cell_matmul_readvariableop_resource3rnn_while_gru_cell_matmul_readvariableop_resource_0"Z
*rnn_while_gru_cell_readvariableop_resource,rnn_while_gru_cell_readvariableop_resource_0"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"�
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2T
(rnn/while/gru_cell/MatMul/ReadVariableOp(rnn/while/gru_cell/MatMul/ReadVariableOp2X
*rnn/while/gru_cell/MatMul_1/ReadVariableOp*rnn/while/gru_cell/MatMul_1/ReadVariableOp2F
!rnn/while/gru_cell/ReadVariableOp!rnn/while/gru_cell/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�K
�
>__inference_rnn_layer_call_and_return_conditional_losses_65813

inputs3
 gru_cell_readvariableop_resource:	�:
'gru_cell_matmul_readvariableop_resource:	�<
)gru_cell_matmul_1_readvariableop_resource:	@�
identity��gru_cell/MatMul/ReadVariableOp� gru_cell/MatMul_1/ReadVariableOp�gru_cell/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masky
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	�*
dtype0s
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num�
gru_cell/MatMul/ReadVariableOpReadVariableOp'gru_cell_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
gru_cell/MatMulMatMulstrided_slice_2:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*(
_output_shapes
:����������c
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split�
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp)gru_cell_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0�
gru_cell/MatMul_1MatMulzeros:output:0(gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*(
_output_shapes
:����������c
gru_cell/ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����e
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split{
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*'
_output_shapes
:���������@_
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:���������@}
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*'
_output_shapes
:���������@c
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:���������@x
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*'
_output_shapes
:���������@t
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*'
_output_shapes
:���������@[
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:���������@m
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������@S
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:���������@l
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:���������@q
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*'
_output_shapes
:���������@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource'gru_cell_matmul_readvariableop_resource)gru_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_65724*
condR
while_cond_65723*8
output_shapes'
%: : : : :���������@: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^gru_cell/MatMul/ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2@
gru_cell/MatMul/ReadVariableOpgru_cell/MatMul/ReadVariableOp2D
 gru_cell/MatMul_1/ReadVariableOp gru_cell/MatMul_1/ReadVariableOp22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_65833

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_65822

inputs
unknown:@

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_64461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_64488
model_input
unknown:	�
	unknown_0:	�
	unknown_1:	@�
	unknown_2:@

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_64475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�

�
(__inference_gru_cell_layer_call_fn_65871

inputs
states_0
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64156o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�
�
%__inference_model_layer_call_fn_64814

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	@�
	unknown_2:@

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_64475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65910

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������@b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������@]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������@Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������@U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������@X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�
�
@__inference_model_layer_call_and_return_conditional_losses_64714

inputs
	rnn_64700:	�
	rnn_64702:	�
	rnn_64704:	@�
dense_64707:@

dense_64709:

identity��dense/StatefulPartitionedCall�rnn/StatefulPartitionedCall�
rnn/StatefulPartitionedCallStatefulPartitionedCallinputs	rnn_64700	rnn_64702	rnn_64704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64671�
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_64707dense_64709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_64461�
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_64472o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp^dense/StatefulPartitionedCall^rnn/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_65723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_65723___redundant_placeholder03
/while_while_cond_65723___redundant_placeholder13
/while_while_cond_65723___redundant_placeholder23
/while_while_cond_65723___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_65569
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_65569___redundant_placeholder03
/while_while_cond_65569___redundant_placeholder13
/while_while_cond_65569___redundant_placeholder23
/while_while_cond_65569___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_64025
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_64025___redundant_placeholder03
/while_while_cond_64025___redundant_placeholder13
/while_while_cond_64025___redundant_placeholder23
/while_while_cond_64025___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
#__inference_rnn_layer_call_fn_65175
inputs_0
unknown:	�
	unknown_0:	�
	unknown_1:	@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_rnn_layer_call_and_return_conditional_losses_64273o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
while_cond_65415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_65415___redundant_placeholder03
/while_while_cond_65415___redundant_placeholder13
/while_while_cond_65415___redundant_placeholder23
/while_while_cond_65415___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
� 
�
while_body_64209
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
while_gru_cell_64231_0:	�)
while_gru_cell_64233_0:	�)
while_gru_cell_64235_0:	@�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
while_gru_cell_64231:	�'
while_gru_cell_64233:	�'
while_gru_cell_64235:	@���&while/gru_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_64231_0while_gru_cell_64233_0while_gru_cell_64235_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_64156r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : �
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@u

while/NoOpNoOp'^while/gru_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ".
while_gru_cell_64231while_gru_cell_64231_0".
while_gru_cell_64233while_gru_cell_64233_0".
while_gru_cell_64235while_gru_cell_64235_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������@: : : : : 2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65949

inputs
states_0*
readvariableop_resource:	�1
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"@   @   ����\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������@:���������@:���������@*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������@M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������@b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������@Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������@]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������@Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������@I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������@U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������@J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������@Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������@V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������@X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:���������:���������@: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0
�
�
#__inference_signature_wrapper_64799
model_input
unknown:	�
	unknown_0:	�
	unknown_1:	@�
	unknown_2:@

	unknown_3:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_63942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������
%
_user_specified_namemodel_input
�
^
B__inference_softmax_layer_call_and_return_conditional_losses_65843

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������
:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
model_input8
serving_default_model_input:0���������;
softmax0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
cell

state_spec"
_tf_keras_rnn_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
C
$0
%1
&2
3
4"
trackable_list_wrapper
C
$0
%1
&2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
,trace_0
-trace_1
.trace_2
/trace_32�
%__inference_model_layer_call_fn_64488
%__inference_model_layer_call_fn_64814
%__inference_model_layer_call_fn_64829
%__inference_model_layer_call_fn_64742�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z,trace_0z-trace_1z.trace_2z/trace_3
�
0trace_0
1trace_1
2trace_2
3trace_32�
@__inference_model_layer_call_and_return_conditional_losses_64991
@__inference_model_layer_call_and_return_conditional_losses_65153
@__inference_model_layer_call_and_return_conditional_losses_64759
@__inference_model_layer_call_and_return_conditional_losses_64776�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z0trace_0z1trace_1z2trace_2z3trace_3
�B�
 __inference__wrapped_model_63942model_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
4iter
	5decay
6learning_rate
7momentum
8rho	rmsq	rmsr	$rmss	%rmst	&rmsu"
	optimizer
,
9serving_default"
signature_map
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

:states
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
@trace_0
Atrace_1
Btrace_2
Ctrace_32�
#__inference_rnn_layer_call_fn_65164
#__inference_rnn_layer_call_fn_65175
#__inference_rnn_layer_call_fn_65186
#__inference_rnn_layer_call_fn_65197�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z@trace_0zAtrace_1zBtrace_2zCtrace_3
�
Dtrace_0
Etrace_1
Ftrace_2
Gtrace_32�
>__inference_rnn_layer_call_and_return_conditional_losses_65351
>__inference_rnn_layer_call_and_return_conditional_losses_65505
>__inference_rnn_layer_call_and_return_conditional_losses_65659
>__inference_rnn_layer_call_and_return_conditional_losses_65813�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zDtrace_0zEtrace_1zFtrace_2zGtrace_3
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
N_random_generator

$kernel
%recurrent_kernel
&bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ttrace_02�
%__inference_dense_layer_call_fn_65822�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zTtrace_0
�
Utrace_02�
@__inference_dense_layer_call_and_return_conditional_losses_65833�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zUtrace_0
:@
2dense/kernel
:
2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
[trace_02�
'__inference_softmax_layer_call_fn_65838�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
�
\trace_02�
B__inference_softmax_layer_call_and_return_conditional_losses_65843�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
&:$	�2rnn/gru_cell/kernel
0:.	@�2rnn/gru_cell/recurrent_kernel
$:"	�2rnn/gru_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_64488model_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_64814inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_64829inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
%__inference_model_layer_call_fn_64742model_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_64991inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_65153inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_64759model_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_64776model_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
�B�
#__inference_signature_wrapper_64799model_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_rnn_layer_call_fn_65164inputs/0"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_65175inputs/0"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_65186inputs"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference_rnn_layer_call_fn_65197inputs"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_65351inputs/0"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_65505inputs/0"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_65659inputs"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
>__inference_rnn_layer_call_and_return_conditional_losses_65813inputs"�
���
FullArgSpecO
argsG�D
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults�

 
p 

 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5
$0
%1
&2"
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
(__inference_gru_cell_layer_call_fn_65857
(__inference_gru_cell_layer_call_fn_65871�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65910
C__inference_gru_cell_layer_call_and_return_conditional_losses_65949�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zftrace_0zgtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_65822inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_65833inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_softmax_layer_call_fn_65838inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_softmax_layer_call_and_return_conditional_losses_65843inputs"�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
h	variables
i	keras_api
	jtotal
	kcount"
_tf_keras_metric
^
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_gru_cell_layer_call_fn_65857inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
(__inference_gru_cell_layer_call_fn_65871inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65910inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65949inputsstates/0"�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
j0
k1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
(:&@
2RMSprop/dense/kernel/rms
": 
2RMSprop/dense/bias/rms
0:.	�2RMSprop/rnn/gru_cell/kernel/rms
::8	@�2)RMSprop/rnn/gru_cell/recurrent_kernel/rms
.:,	�2RMSprop/rnn/gru_cell/bias/rms�
 __inference__wrapped_model_63942t&$%8�5
.�+
)�&
model_input���������
� "1�.
,
softmax!�
softmax���������
�
@__inference_dense_layer_call_and_return_conditional_losses_65833\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������

� x
%__inference_dense_layer_call_fn_65822O/�,
%�"
 �
inputs���������@
� "����������
�
C__inference_gru_cell_layer_call_and_return_conditional_losses_65910�&$%\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p 
� "R�O
H�E
�
0/0���������@
$�!
�
0/1/0���������@
� �
C__inference_gru_cell_layer_call_and_return_conditional_losses_65949�&$%\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p
� "R�O
H�E
�
0/0���������@
$�!
�
0/1/0���������@
� �
(__inference_gru_cell_layer_call_fn_65857�&$%\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p 
� "D�A
�
0���������@
"�
�
1/0���������@�
(__inference_gru_cell_layer_call_fn_65871�&$%\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������@
p
� "D�A
�
0���������@
"�
�
1/0���������@�
@__inference_model_layer_call_and_return_conditional_losses_64759p&$%@�=
6�3
)�&
model_input���������
p 

 
� "%�"
�
0���������

� �
@__inference_model_layer_call_and_return_conditional_losses_64776p&$%@�=
6�3
)�&
model_input���������
p

 
� "%�"
�
0���������

� �
@__inference_model_layer_call_and_return_conditional_losses_64991k&$%;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������

� �
@__inference_model_layer_call_and_return_conditional_losses_65153k&$%;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������

� �
%__inference_model_layer_call_fn_64488c&$%@�=
6�3
)�&
model_input���������
p 

 
� "����������
�
%__inference_model_layer_call_fn_64742c&$%@�=
6�3
)�&
model_input���������
p

 
� "����������
�
%__inference_model_layer_call_fn_64814^&$%;�8
1�.
$�!
inputs���������
p 

 
� "����������
�
%__inference_model_layer_call_fn_64829^&$%;�8
1�.
$�!
inputs���������
p

 
� "����������
�
>__inference_rnn_layer_call_and_return_conditional_losses_65351�&$%S�P
I�F
4�1
/�,
inputs/0������������������

 
p 

 

 
� "%�"
�
0���������@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_65505�&$%S�P
I�F
4�1
/�,
inputs/0������������������

 
p

 

 
� "%�"
�
0���������@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_65659q&$%C�@
9�6
$�!
inputs���������

 
p 

 

 
� "%�"
�
0���������@
� �
>__inference_rnn_layer_call_and_return_conditional_losses_65813q&$%C�@
9�6
$�!
inputs���������

 
p

 

 
� "%�"
�
0���������@
� �
#__inference_rnn_layer_call_fn_65164t&$%S�P
I�F
4�1
/�,
inputs/0������������������

 
p 

 

 
� "����������@�
#__inference_rnn_layer_call_fn_65175t&$%S�P
I�F
4�1
/�,
inputs/0������������������

 
p

 

 
� "����������@�
#__inference_rnn_layer_call_fn_65186d&$%C�@
9�6
$�!
inputs���������

 
p 

 

 
� "����������@�
#__inference_rnn_layer_call_fn_65197d&$%C�@
9�6
$�!
inputs���������

 
p

 

 
� "����������@�
#__inference_signature_wrapper_64799�&$%G�D
� 
=�:
8
model_input)�&
model_input���������"1�.
,
softmax!�
softmax���������
�
B__inference_softmax_layer_call_and_return_conditional_losses_65843\3�0
)�&
 �
inputs���������


 
� "%�"
�
0���������

� z
'__inference_softmax_layer_call_fn_65838O3�0
)�&
 �
inputs���������


 
� "����������
