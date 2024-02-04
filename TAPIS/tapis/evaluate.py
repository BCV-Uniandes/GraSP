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
    parser.add_argument('--coco-ann-path', default=None, type=str, help='path to coco style anotations')
    parser.add_argument('--pred-path', default=None, type=str, help='path to predictions')
    parser.add_argument('--filter', action='store_true', help='path to predictions')
    parser.add_argument('--tasks', nargs='+', help='tasks to be evaluated', required=True, default=None)
    parser.add_argument('--metrics', nargs='+', help='metrics to be evaluated',
                        choices=['mAP', 'mAP@0.5IoU_box', 'mAP@0.5IoU_segm', 'mIoU', 'mIoU_mAP@0.5',
                                 'classification', 'detection','inst_segmentation', 
                                 'sem_segmentation', 'segmentation'],
                        required=True, default=None)
    parser.add_argument('--masks-path', default=None, type=str, help='path to predictions')
    parser.add_argument('--selection', type=str, default='thresh', 
                        choices=['thresh', 'topk', 'topk_thresh', 'cls_thresh', 'cls_topk', 'cls_topk_thresh', 'all'], 
                        help='Prediction selection method')
    parser.add_argument('--selection_info', help='Hypermarameters to perform selection', default=0.75)
    parser.add_argument('--output_path', default=None, type=str, help='path to predictions')

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