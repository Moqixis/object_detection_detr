## [DETR](https://github.com/facebookresearch/detr)训练自己的数据集

### 1.参考教程&代码

☑如何用DETR（detection transformer）训练自己的数据集 [🔗](https://blog.csdn.net/weixin_50233398/article/details/121785953)   [🔗](https://github.com/DataXujing/detr_transformer)

【voc2cc参考】windows10复现DEtection TRansformers（DETR）并实现自己的数据集  [🔗](https://blog.csdn.net/w1520039381/article/details/118905718)

### 2.训练mAP一直为0的问题

detr.py里面的build()函数，num_classes的值，全是 真实class数量+1，下载原作者的代码看看，里面有注释。

```python
def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 1+1 if args.dataset_file != 'coco' else 1+1 # <--------修改
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 1+1           # <--------修改
    device = torch.device(args.device)
```

### 3.主函数

运行main.py的时候，注意里面好多参数没设置default值

### 4.其他

徐静的github代码：inference_video.py里面，file标红，可以改成str(i)+".jpg"，会按帧存下来

