# pip install xmltodict
import json
import xmltodict
from glob import glob
from os import path as osp
from pathlib import Path
import cv2
import numpy as np
from shutil import copy2
from PIL import Image

"""
Input Dataset Structure

BASE_DIR
├── SM0915_13
│   ├── 27_SM0915_13.xml
│   ├── MP_SEL_SUR_001441.jpg
│   ├── MP_SEL_SUR_001442.jpg
│   ├── MP_SEL_SUR_001443.jpg
│   ├── ...
├── SM0915_14
│   ├── 28_SM0915_14.xml
│   ├── MP_SEL_SUR_001562.jpg
│   ├── MP_SEL_SUR_001563.jpg
│   ├── MP_SEL_SUR_001564.jpg
│   ├── ...
├── ...
└── color_index.xlsx

Output Dataset Structure

OUTPUT_DIR
├── annotations
│   ├── *.xml
├── images
│   ├── *.jpg
├── masks
│   ├── *.png
└── color_index.xlsx
"""

### RUN OPTIONS ###
BASE_DIR = Path('/home/super/Projects/dataset/surface_org')
XML_GLOB = Path(BASE_DIR) / '**/*.xml'
OUTPUT_DIR = Path('/home/super/Projects/dataset/surface6')
JSON_OUTPUT = OUTPUT_DIR / 'annotations'
IMAGE_OUTPUT = OUTPUT_DIR / 'images'
MASK_OUTPUT = OUTPUT_DIR / 'masks'

# [0, 0, 0] for ignore mask
color_map = {
"sidewalk@blocks": [0, 255, 0],  # sidewalk
"sidewalk@cement": [0, 255, 0],  # sidewalk
"sidewalk@urethane": [255, 128, 0],  # bike_lane
"sidewalk@asphalt": [255, 128, 0],  # bike_lane
"sidewalk@soil_stone": [0, 255, 0],  # sidewalk
"sidewalk@damaged": [0, 255, 0],  # sidewalk
"sidewalk@other": [0, 255, 0],  # sidewalk
"braille_guide_blocks@normal": [255, 255, 0],  # guide_block
"braille_guide_blocks@damaged": [255, 255, 0],  # guide_block
"roadway@normal": [0, 0, 255],  # roadway
"roadway@crosswalk": [255, 0, 255],  # crosswalk
"alley@normal": [0, 0, 255],  # roadway
"alley@crosswalk": [255, 0, 255],  # crosswalk
"alley@speed_bump": [0, 0, 255],  # roadway
"alley@damaged": [0, 0, 255],  # roadway
"bike_lane@normal": [255, 128, 0],  # bike_lane
"caution_zone@stairs": [255, 0, 0],  # caution_zone
"caution_zone@manhole": [0, 0, 0],  # background
"caution_zone@tree_zone": [255, 0, 0],  # caution_zone
"caution_zone@grating": [255, 0, 0],  # caution_zone
"caution_zone@repair_zone": [255, 0, 0],  # caution_zone
}
###################

error_logs = []
image2path = {}
statistics = {}

def convert_to_json():
    xml_files = sorted(glob(str(XML_GLOB), recursive=True))
    JSON_OUTPUT.mkdir(exist_ok=False, parents=True)

    for i, xml in enumerate(xml_files):
        xml = Path(xml)
        out_json = JSON_OUTPUT / xml.name.replace('.xml', '.json')
        print(f'[{i+1}/{len(xml_files)}] Converting {xml} -> {out_json}')

        with open(xml, 'r') as f:
            xml_string = f.read()

        xml_dict = xmltodict.parse(xml_string)
        json_string = json.dumps(xml_dict, indent=4)

        with open(out_json, 'w+') as f:
            f.write(json_string)

def index_images():
    global image2path
    for image in BASE_DIR.rglob('*.jpg'):
        image2path[image.name] = image
    print(f"Found {len(list(image2path.keys()))} images from dataset.")

def add_count(key: str):
    if key not in statistics:
        statistics[key] = 0
    statistics[key] += 1

def draw_polygon(points: list, mask: np.array, rgb):
    pts = np.asarray(points).reshape((-1, 1, 2))
    r, g, b = rgb
    mask = cv2.fillPoly(mask, [pts], color=(r,g,b))
    return mask

def parse_polygon(polygon: dict, mask: np.array):
    label = polygon['@label']  # str 'sidewalk'
    # occluded = polygon['@occluded']  # str '0' I think this is always zero
    points = str(polygon['@points'])  # str '778.28,0.00;0.56,1080.00;1920.00,1080.00;...'
    # z_order = polygon['@z_order']  # str '2'

    if 'attribute' in polygon:
        attribute = polygon['attribute']  # dict
        # attribute_name = attribute['@name']  # str 'attribute' useless...
        attribute_text = attribute['#text']  # str 'blocks', 'normal', ...'
    else:
        attribute_text = 'normal'

    cls = str(label) + '@' + str(attribute_text)
    assert cls in color_map, f"Invalid label@attribute: {cls}"
    add_count(cls)

    rgb = color_map[cls]
    if rgb == [0, 0, 0]:  # Ignore polygon
        return mask

    xy_points = [[round(float(x)), round(float(y))] for x, y in [str(xy_str).split(',') for xy_str in points.split(';')]]
    mask = draw_polygon(xy_points, mask, rgb)
    return mask

def parse_image(image: dict, json_file: Path):
    global image2path

    id = image['@id']  # str '0'
    name = str(image['@name'])  # str 'MP_SEL_SUR_007985.jpg'
    width = int(image['@width'])  # str '1920'
    height = int(image['@height'])  # str '1080'

    if 'polygon' in image:
        polygon_list = image['polygon']  # list or dict(if only one instance)
    elif 'polyline' in image:
        polygon_list = image['polyline']
    else:
        raise KeyError(f"Couldn't find polygon key from dict.")

    ### Masking polygon ###
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    if isinstance(polygon_list, dict):
        polygon_list = [polygon_list]
    polygon_list = sorted(polygon_list, key=lambda poly: int(poly['@z_order']))
    for polygon in polygon_list:
        mask = parse_polygon(polygon, mask)

    ### Saving mask ###
    mask_file = MASK_OUTPUT / str(name).replace('.jpg', '.png')
    Image.fromarray(mask).save(mask_file)
    add_count('mask')

    ### Saving image ###
    if not name in image2path:
        raise FileNotFoundError(f"Cannot find {name} from indexed image paths.")
    copy2(image2path[name], IMAGE_OUTPUT)
    add_count('image')

def parse_json(json_file, **kwargs):
    # root > annotations > image[] > {'@id', '@name', '@width', '@height', 'polygon[]'}
    global error_logs

    json_file = Path(json_file)
    with open(json_file, 'r') as f:
        ann = json.load(f)
    annotations = ann['annotations']
    # version = annotations['version']  # str
    # meta = annotations['meta']  # dict
    image_list = annotations['image']  # list

    for i, image in enumerate(image_list):
        print(f"[{kwargs['json_count']}/{kwargs['json_total']}] [{i + 1}/{len(image_list)}] {image['@name']}")
        try:
            parse_image(image, json_file)
        except KeyError as e:
            print(e)
            error_logs.append(f"{json_file.name}-{image['@name']}-{e}")

def generate_masks():
    global error_logs

    MASK_OUTPUT.mkdir(exist_ok=False, parents=True)
    IMAGE_OUTPUT.mkdir(exist_ok=False, parents=True)

    index_images()

    json_files = sorted(glob(str(JSON_OUTPUT / '**/*.json'), recursive=True))
    for i, json_file in enumerate(json_files):
        print(f'[{i+1}/{len(json_files)}] {json_file}')
        parse_json(json_file, json_count=i+1, json_total=len(json_files))

    print(f'Error count: {len(error_logs)}')
    print(error_logs)
    print(statistics)

if __name__ == '__main__':
    if not JSON_OUTPUT.exists():
        convert_to_json()

    generate_masks()
