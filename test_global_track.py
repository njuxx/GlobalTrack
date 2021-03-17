import _init_paths
import neuron.data as data
from trackers import *


if __name__ == '__main__':
    cfg_file = 'configs/qg_rcnn_r50_fpn.py'
    ckp_file = 'checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
    transforms = data.BasicPairTransforms(train=False)
    tracker = GlobalTrack(
        cfg_file, ckp_file, transforms,
        name_suffix='qg_rcnn_r50_fpn')
    evaluators = [
        # data.EvaluatorOTB(version=2015, root_dir="/disk/xuxiang/GlobalTrack/data/OTB100"),
        # data.EvaluatorLaSOT(frame_stride=10, root_dir="/disk/xuxiang/GlobalTrack/data/LaSOTBenchmark"),
        # data.EvaluatorGOT10k(subset='test', root_dir="/disk/xuxiang/GlobalTrack/data/GOT-10k"),
        data.EvaluatorTLP(root_dir="/disk/xuxiang/GlobalTrack/data/TLP")
    ]
    for e in evaluators:
        e.run(tracker, visualize=False)
        e.report(tracker.name)
