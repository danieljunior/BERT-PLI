# -*- coding: utf-8 -*-
__author__ = 'yshao'

import os
import argparse
import json
import logging
from tqdm import tqdm

from provenance.retrospective_service import RetrospectiveService
from provenance.prospective_service import ProspectiveService

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def load_json_lines(file_path):
    """Load JSON lines file into a dictionary"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if 'id_' in item:
                data[item['id_']] = item
            else:
                data[item['guid']] = item
    return data

def process_files(in_file_path, out_file_path, result_file_path):
    """Process input and output files to generate result file"""
    # Load both files
    logger.info(f"Loading input file: {in_file_path}")
    in_data = load_json_lines(in_file_path)
    
    logger.info(f"Loading output file: {out_file_path}")
    out_data = load_json_lines(out_file_path)
    
    # Process and write results
    logger.info(f"Processing and writing results to: {result_file_path}")
    with open(result_file_path, 'w', encoding='utf-8') as f:
        for guid, in_item in tqdm(in_data.items(), desc="Processing items"):
            if guid in out_data:
                result = {
                    "guid": guid,
                    "res": out_data[guid]["res"],
                    "label": in_item["label"]
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                logger.warning(f"No matching output found for guid: {guid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paras-file', '-in', help="input paragraphs file", required=True)
    parser.add_argument('--poolout-file', '-out', help="poolout file", required=True)
    parser.add_argument('--result', help="result file path", required=True)
    parser.add_argument('--test', help="test mode", action='store_true')

    args = parser.parse_args()
    
    dataflow_tag = os.getenv('DATAFLOW_TAG', ProspectiveService.DEFAULT_DATAFLOW_TAG)
    provenance = RetrospectiveService(dataflow_tag, bypass=True)
    input_data = {}
    if args.test:
        task = ProspectiveService.TF_TEST_PARSE_POOLOUT
        result_key = ProspectiveService.DT_TEST_PARSED_POOLOUT_DATA
    else:
        task = ProspectiveService.TF_TRAIN_PARSE_POOLOUT
        result_key = ProspectiveService.DT_TRAIN_PARSED_POOLOUT_DATA
    with provenance.get_retrospective_data(task, input_data) as result:
        if not os.path.exists(args.result):
            process_files(args.paras_file, args.poolout_file, args.result)
        else:
            print(f"Output file already exists. Exiting.")

        result[result_key] = [[args.result]]

    logger.info("Processing completed")
