# pip install xmltodict
import json
import xmltodict
from glob import glob
from os import path as osp
from pathlib import Path
import cv2

ROOT_DIR = Path('/home/super/Projects/dataset/surface')
XML_GLOB = Path(ROOT_DIR) / 'annotations/*.xml'
JSON_OUTPUT = Path(ROOT_DIR) / 'annotations_json'

def convert_to_json():
    xml_files = list(XML_GLOB.rglob(XML_GLOB))
    JSON_OUTPUT.mkdir(exist_ok=False)

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

def parse_json(json_file):
    pass

def generate_masks():
    json_files = list(JSON_OUTPUT.rglob('*.json'))

    for i, json_file in enumerate(json_files):
        parse_json(json_file)


if __name__ == '__main__':
    if not JSON_OUTPUT.exists():
        convert_to_json()

