import json
import os
import argparse
import sys

import cv2

# import detectron.utils.boxes as box_utils
from detectron2.structures import BoxMode

# output format:
# <frame_index> <ILSVRC2015_VID_ID> <confidence> <xmin> <ymin> <xmax> <ymax>


def xywh_to_xyxy(bbox):
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    return [e.item() for e in bbox]

def det_to_vid(detection, frame_id_map, vid_val_data):
    # from: /datasets/vid/ILSVRC/Data/VID/val/ILSVRC2015_val_00051001/000000.JPEG
    # to: 'ILSVRC2015_val_00007003/000000'
    frame_id = detection['image_id']
    im = vid_val_data['images'][frame_id]
    assert frame_id == im['id']

    frame_id = im['file_name'].replace('/datasets/vid/ILSVRC/Data/VID/val/', '')
    frame_id = frame_id.split('.')[0]

    out  = ' '.join(map(str, [frame_id_map[frame_id], detection['category_id']+1, detection['score']]))
    out += ' '
    bbox = xywh_to_xyxy(detection['bbox'])
    out += ' '.join(map(str, bbox))
    return out, '/datasets/vid/ILSVRC/Data/VID/val/' + frame_id + '.JPEG'


def generate_img(img_path, bbox, idx, text):
    bbox = xywh_to_xyxy(bbox)
    bbox = list(map(int, bbox))
    img = cv2.imread(img_path)
    img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2], bbox[3]),(255,0,0),3)
    cv2.putText(img, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imwrite('imgs/{}.png'.format(idx), img)

def main():
    args = parse_args()
    dets_file = args.dets_file[0]
    vid_val = args.vid_val[0]
    vid_val_json = args.vid_val_json[0]

    output_dir = os.path.dirname(dets_file)
    output_name = '{}_vid_vidformat.txt'.format(os.path.basename(dets_file).split('.')[0])
    output_path = os.path.join(output_dir, output_name)

    with open(dets_file) as f:
        bbox_data = json.load(f)

    with open(vid_val_json) as f:
        vid_val_data = json.load(f)
    
    frame_id_map = {}
    with open(vid_val) as f:
        for line in f:
            frame_id, seq = line.split()
            frame_id_map[frame_id] = seq
    

    with open(output_path, 'w') as f:
        for idx, bbox in enumerate(bbox_data):
            line, img_path = det_to_vid(bbox, frame_id_map, vid_val_data)
            # if bbox['score'] > 0.7:
            #     generate_img(img_path, map(int,bbox['bbox']), idx, '{}({})'.format(bbox['category_id'], bbox['score']))
            f.write("%s\n" % line)


def parse_args():
    parser = argparse.ArgumentParser(description='COCO detections output to ImageNet VID')

    parser.add_argument(
        'vid_val',
        nargs=1,
        help='ILSVRC/ImageSets/VID/val.txt',
        type=str
    )
    parser.add_argument(
        'vid_val_json',
        nargs=1,
        help='vid_val_coco_format.json in results directory',
        type=str
    )
    parser.add_argument(
        'dets_file',
        nargs=1,
        help='detections json file in COCO format',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    main()
