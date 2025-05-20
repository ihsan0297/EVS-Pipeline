import sys
import pathlib
from fastapi import FastAPI, HTTPException
from database_handler import DatabaseHandler
from custom_transforms import TimmRandAugmentTransform  # ✅ Import from external module
import os
import os
import logging
from PIL import Image
import pandas as pd 

# -----------------------------
# GLOBAL CONFIG & SETTINGS
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",         # Log output goes to this file
    filemode="a"                # Append to the file (use "w" to overwrite each time)
)


# Settings for inference

from rabbitmq_handler import RMQHandler
# Fix WindowsPath issue if running on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from qrupdate import process_images
import enums
from utils import*
# ✅ Register the transform in the global namespace
sys.modules['custom_transforms.TimmRandAugmentTransform'] = TimmRandAugmentTransform

# ✅ Also register it in the built-in __main__ module

from Model import Model



app = FastAPI()
model = Model()
db=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.MYSQL_DATABASE)
db_barcode=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.DATABASE_FAULT_APP)
rmq=RMQHandler(enums.RMQ_HOST, enums.RMQ_USER,enums.RMQ_PASSWORD)


@app.on_event("startup")
def startup_event():
    print("Loading models...")
    model.load_models()
    print("Models loaded successfully!")
    
        



# from logging_config import LOGGING_CONFIG
# import logging.config
# logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)  # this will now write into app.log
import json
from similarity_test import annotate_with_similarity

import io
import sys
import contextlib

@app.get("/inference_image")
def inference_images():
    try:
        logger.info("Starting inference_images")

        if not db.connect():
            logging.error("DB Connection Failed")
            return {"status": "error", "message": "Database Connection Failed"}

        if not rmq.rabbitMQConnection():
            logging.error("RMQ Connection Failed")
            return {"status": "error", "message": "RMQ Connection Failed"}
        if not db_barcode.connect():
            logging.error("DB Connection Failed")
            return {"status": "error", "message": "Database Connection Failed"}
        

        message_bundle = rmq.fetch_messages()
        messages = message_bundle.get("images", [])
        raw_messages = message_bundle.get("raw", [])

        if not messages:
            return {"status": "error", "message": "No images found in queue"}

        # Group messages by roll_header_id
        roll_groups = {}
        for msg in messages:
            roll_id = msg.get(enums.ROLL_HEADER_ID_COLUMN)
            if roll_id is not None:
                roll_groups.setdefault(roll_id, []).append(msg)

        all_combined_results = []

        for roll_id, roll_msgs in roll_groups.items():
            print(f"🎯 Processing Roll Header ID: {roll_id}")
            logging.info("Processing Roll Header ID: %s", roll_id)

            # Fetch previous sticker model results
            prev_res=db.get_next_greater_row_by_roll_header_id(roll_id)
            
            print("Previous result:", prev_res)
            
            # prev_res = db.fetch_latest_sticker_model(roll_id)
            all_prev_results_df = pd.DataFrame()
            if prev_res is not None and not prev_res.empty:
                all_prev_results_df = prev_res
                print("✅ Previous sticker model found for roll_id:", roll_id)
            else:
                print("⚠️ No sticker model found for roll_id:", roll_id)

            image_paths = [msg[enums.IMAGE_PATH_COLUMN] for msg in roll_msgs]

            seam_results = model.process_images_seam_model(image_paths)
            seam_result_map = {
                res["image_path"]: {
                    "final_class": res.get("final_class"),
                    "conf": round(res.get("confidence", 0), 4)
                }
                for res in seam_results
            }

            combined_results = []
            qr_candidates = set()

            for idx, msg in enumerate(roll_msgs):
                image_path = msg[enums.IMAGE_PATH_COLUMN]
                combined = {
                    enums.IMAGE_TITLE_COLUMN: os.path.basename(image_path),
                    enums.IMAGE_PATH_COLUMN: image_path,
                    enums.METERS_COLUMN: msg.get(enums.METERS_COLUMN),
                    enums.ROLL_HEADER_ID_COLUMN: roll_id,
                    enums.IS_BARCODE_READABLE_COLUMN: msg.get(enums.IS_BARCODE_READABLE_COLUMN),
                    "seam_data": seam_result_map.get(image_path)
                }

                seam_pred = seam_result_map.get(image_path, {}).get("final_class")
                if seam_pred == enums.SEAM:
                    print('Seam detected in image:', image_path)
                #     start = max(0, idx - enums.LOOKUP_DOWN_IMAGE_COUNT)
                #     end = min(len(roll_msgs), idx + enums.LOOKUP_DOWN_IMAGE_COUNT + 1)
                #     for i in range(start, end):
                #         if i != idx:
                #             qr_candidates.add(roll_msgs[i][enums.IMAGE_PATH_COLUMN])

                combined_results.append(combined)

            # Sort QR candidates by meters before sending to QR model
            qr_candidates_info = [
                msg for msg in roll_msgs if msg[enums.IMAGE_PATH_COLUMN] in qr_candidates
            ]
            qr_candidates_sorted = sorted(
                combined_results,
                key=lambda x: x.get(enums.METERS_COLUMN, float('inf'))
            )
            qr_paths_sorted = [msg[enums.IMAGE_PATH_COLUMN] for msg in qr_candidates_sorted]

            # Run QR detection
            print("Running QR detection on images...")
            with contextlib.redirect_stdout(io.StringIO()):
                qr_results = process_images(qr_paths_sorted)
            # qr_results = process_images(qr_paths_sorted)
            print("QR detection Finished.....")
            with open("qr.json", "w", encoding="utf-8") as f:
                json.dump(qr_results, f, indent=2, ensure_ascii=False)
            #Similarity Test
            try:
                print('...............Starting similarity test............')
                barcode_from_db=db_barcode.fetch_barcode_by_header_id(roll_id)
                # barcode_values = [b[0] for b in barcode_from_db if b[0]]
                
                if len(barcode_from_db)<=0:
                    logging.error("❌ No barcodes found in the database")
                    return {"status": "error", "message": "No barcodes found in the database"}

                qr_items = qr_results.get('qr_Res', [])
                sanitized_items = []
                
                for item in qr_items:
                    if item is None:
                        continue
                        
                    sanitized_item = {}
                    sanitized_item["image_path"] = item.get("image_path", "")
                    
                    # Ensure detection is a dictionary
                    if item.get("detection") is None:
                        sanitized_item["detection"] = {}
                    else:
                        sanitized_item["detection"] = item["detection"]
                        
                    # Ensure fast_qr exists (can be None)
                    sanitized_item["fast_qr"] = item.get("fast_qr")
                    
                    # Ensure ocr is a dictionary
                    if item.get("ocr") is None:
                        sanitized_item["ocr"] = {}
                    else:
                        sanitized_item["ocr"] = item["ocr"]
                        
                    sanitized_items.append(sanitized_item)
                
                # Use sanitized data
                sanitized_qr_results = {"qr_Res": sanitized_items}

                qr_items = qr_results.get('qr_Res', [])
                ultra_processed_barcodes = annotate_with_similarity(
                     sanitized_qr_results, barcode_from_db, top_n=2
                )
                # print("Ultra processed barcodes:", ultra_processed_barcodes)
                with open("qr_sim.json", "w", encoding="utf-8") as f:
                    json.dump(ultra_processed_barcodes, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Error in similarity test: {e}")
                return {"status": "error", "message": str(e)}
            try:
                qr_data_map = {}
                if ultra_processed_barcodes and "qr_Res" in ultra_processed_barcodes:
                    for item in ultra_processed_barcodes["qr_Res"]:
                        if item and "image_path" in item:
                            qr_data_map[item["image_path"]] = item
        # Merge QR data into combined results    
                for result in combined_results:
                    path = result[enums.IMAGE_PATH_COLUMN]
                    
                    if path in qr_data_map:
                        result["qr_data"] = qr_data_map[path]
                apply_checks(combined_results, prev_res, raw_messages,roll_id,rmq,db)
                meters_vals = [res.get(enums.METERS_COLUMN) for res in combined_results]

                if not meters_vals or meters_vals[0] is None or meters_vals[-1] is None:
                    raise ValueError("Meters column is missing or empty – cannot name results file")

                first_meters = meters_vals[0]
                last_meters  = meters_vals[-1]
                out_dir = pathlib.Path("output")
                out_dir.mkdir(exist_ok=True)
                results_path = out_dir / f"results_{first_meters}_{last_meters}.json"
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump({"images": combined_results}, f, indent=2, ensure_ascii=False)
                print('...............Finished similarity test............')
            except Exception as e:
                print(f"❌ Error in Combining Results: {e}")
                return {"status": "error", "message": str(e)}
            # save_results_to_json({"images": combined_results}, f"results_{roll_id}.json")

            all_combined_results.extend(combined_results)

        return {"status": "success", "message": f"Processed {len(all_combined_results)} images across {len(roll_groups)} rolls"}

    except Exception as e:
        logging.error("Error in inference_images: %s", e)
        print(f"❌ Error in inference_images: {e}")
        return {"status": "error", "message": str(e)}
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)