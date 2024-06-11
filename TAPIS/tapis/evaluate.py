import argparse
import os
import json
import pandas as pd
import numpy as np
from evaluate.main_eval import eval_task
from evaluate.utils import load_json, read_detectron2_output

def main(coco_ann_path, pred_path, tasks, metrics, output_dir, sufix, masks_path):
    # Load coco anns and preds
    coco_anns = load_json(coco_ann_path)
    preds = load_json(pred_path) if type(pred_path)==str else pred_path
    all_metrics = {}
    for task, metric in zip(tasks,metrics):
        task_eval, aux_metrics  = eval_task(task, metric, coco_anns, preds, masks_path)
        aux_metrics = dict(zip(aux_metrics.keys(),map(lambda x: round(x,8), aux_metrics.values())))
        print('{} task {}: {} {}'.format(task, metric, round(task_eval,8), aux_metrics))
        final_metrics = {metric: round(task_eval,8)}
        final_metrics.update(aux_metrics) 
        all_metrics[task] = final_metrics

        if output_dir is not None and sufix is not None:
            if metric in ['mAP@0.5IoU_box','detection']:
                met_suf = 'det'
            elif metric in ['mAP@0.5IoU_segm','inst_segmentation']:
                met_suf = 'ins_seg'
            elif metric in ['mIoU', 'sem_segmentation']:
                met_suf = 'sem_seg'
            elif metric=='mIoU_mAP@0.5':
                met_suf = 'seg'
            else:
                met_suf = 'class'
            if os.path.isfile(os.path.join(output_dir,f'metrics_{met_suf}.json')):
                with open(os.path.join(output_dir,f'metrics_{met_suf}.json'),'r') as f:
                    save_json = json.load(f)
                    save_json[sufix] = all_metrics[task]
            else:
                save_json = {sufix:all_metrics[task]}
            with open(os.path.join(output_dir,f'metrics_{met_suf}.json'),'w') as f:
                json.dump(save_json,f, indent=4)

            excel_file = os.path.join(output_dir,f'metrics_{met_suf}.xlsx')
            if os.path.exists(excel_file):
                existing_df = pd.read_excel(excel_file)
            else:
                existing_df = pd.DataFrame()

            # breakpoint()
            new_row = {k: v for k, v in [('Experiments',sufix)]+list(all_metrics[task].items()) if k=='Experiments' or v>0}
            new_df = pd.DataFrame([new_row])

            updated_df = existing_df.append(new_df, ignore_index=True)
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                updated_df.to_excel(writer, index=False)
    overall_metric = np.mean([v[m] for v,m in zip(list(all_metrics.values()), metrics)])
    print('Overall Metric: {}'.format(overall_metric))
    return overall_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation parser')
    parser.add_argument('--coco-ann-path', default=None, type=str, required=True, help='Path to coco anotations')
    parser.add_argument('--pred-path', default=None, type=str, required=True, help='Path to predictions')
    parser.add_argument('--filter', action='store_true', help='Filter predictions')
    parser.add_argument('--tasks', nargs='+', help='Tasks to be evaluated', default=None, required=True)
    parser.add_argument('--metrics', nargs='+', help='Metrics to be evaluated',
                        choices=['mAP', # Classification mean Average Precision
                                 'mAP@0.5IoU_box', # Detection mean Average Precision with 0.5 bounding box IoU threshold
                                 'mAP@0.5IoU_segm', # Instance Segmentation mean Average Precision with 0.5 mask IoU threshold
                                 'mIoU', # Semantic segmentation mean Intersection over Union
                                 'mIoU_mAP@0.5', # Semantic segmentation IoU and instance segmentation mean Average Precision with a 0.5 mask threshold
                                 'classification', # Same as 'mAP' (Classification mean Average Precision)
                                 'detection', # Same as 'mAP@IoU_box' (Detection mean Average Precision with 0.5 bounding box IoU threshold)
                                 'inst_segmentation', # Same as 'mAP@0.5IoU_segm' (Instance Segmentation mean Average Precision with 0.5 mask IoU threshold)
                                 'sem_segmentation', # Same as 'mIoU' (Semantic segmentation mean Intersection over Union)
                                 'segmentation' # Same as 'mIoU_mAP@0.5' (Semantic segmentation IoU and instance segmentation mean Average Precision with a 0.5 mask threshold)
                                 ],
                        default=None,
                        required=True)
    parser.add_argument('--masks-path', type=str, required=False, help='Path to semantic segmentation ground truth images')
    parser.add_argument('--selection', type=str, default='thresh', 
                        choices=['thresh', # General threshold filtering
                                'topk', # General top k filtering
                                'topk_thresh', # Threshold and top k filtering
                                'cls_thresh', # Per-class threshold filtering
                                'cls_topk', # Per-class top k filtering
                                'cls_topk_thresh', # Per-class top k and and threshold filtering
                                'all' # No filtering
                                ], 
                        required=False,
                        default=None,
                        help='Prediction filtering method')
    parser.add_argument('--selection_info', help='Hypermarameters to perform filtering', required=False, default=0.75)
    parser.add_argument('--output_path', default=None, type=str, help='Output directory')

    args = parser.parse_args()
    print(args)
    
    assert len(args.tasks) == len(args.metrics), f'{args.tasks} {args.metrics}'
    preds = args.pred_path
    if args.filter:
        assert len(args.metrics)==1, args.metrics
        assert args.tasks == ['instruments'], args.tasks
        segmentation = 'segmentation' in args.metrics[0] or 'seg' in args.metrics[0] or 'mIoU' in args.metrics[0]
        if args.selection =='thresh':
            selection_params = [None, float(args.selection_info)]
        elif args.selection == 'topk':
            selection_params = [int(args.selection_info), None]
        elif args.selection == 'topk_thresh':
            assert type(args.selection_info) == str and ',' in args.selection_info and len(args.selection_info.split(','))==2
            selection_params = args.selection_info.split(',')
            selection_params[0] = int(selection_params[0])
            selection_params[1] = float(selection_params[1])
        elif 'cls' in args.selection:
            assert type(args.selection_info) == str
            assert os.path.isfile(args.selectrion_info)
            with open(args.selection_info, 'r') as f:
                selection_params = json.load(f)
        else:
            raise ValueError(f'Incorrect selection type {args.selection}')
        preds = read_detectron2_output(args.coco_ann_path, preds, args.selection, selection_params, segmentation)
    
    output_dir = None
    sufix = None
    if args.output_path is not None:
        if args.selection in ['thresh','topk','topk_thresh']:
            sufix = f"{args.selection}_{args.selection_info}"
        elif 'cls' in args.selection:
            sufix = args.selection_info.split('/')[-1].replace('.json','')
        elif 'all' == args.selection:
            sufix = 'all'
        else:
            breakpoint()
        output_dir = args.output_path
    main(args.coco_ann_path, preds, args.tasks, args.metrics, output_dir, sufix, args.masks_path)