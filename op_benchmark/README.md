description
====
>this sample compare some simple op between tf/xla and tvm.  

op
====
### batch_normalization
>usage:
```
python3 tvm_op_autosheduler.py --op batch_normalization
python3 xla_op.py --op batch_normalization
```
### normalization
>usage:
```
python3 tvm_op_autosheduler.py --op normalization --axis 1
python3 xla_op.py --op normalization -axis 1
```
### reduce
>usage:
```
python3 tvm_op_autosheduler.py --op  reduce --axis 0
python3 xla_op.py --op reduce --axis 0
```
### element_wise
>usage:
```
python3 tvm_op_autosheduler.py --op element_wise
python3 xla_op.py --op element_wise
```
