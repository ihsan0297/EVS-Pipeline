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
import enums
# from utils import*
# ✅ Register the transform in the global namespace
sys.modules['custom_transforms.TimmRandAugmentTransform'] = TimmRandAugmentTransform

# ✅ Also register it in the built-in __main__ module

from Model import Model



app = FastAPI()
model = Model()
db=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.MYSQL_DATABASE)
# db_barcode=DatabaseHandler(enums.MYSQL_HOST,enums.MYSQL_USER_NAME,enums.MYSQL_PASSWORD,enums.DATABASE_FAULT_APP)
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
# from similarity_test import annotate_with_similarity

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
        # if not db_barcode.connect():
        #     logging.error("DB Connection Failed")
        #     return {"status": "error", "message": "Database Connection Failed"}
        

        message_bundle = rmq.fetch_messages()
        messages = message_bundle.get("images", [])
        delivery_tags = message_bundle.get("raw", [])
        # delivery_tags = message_bundle.get("delivery_tags", [])

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
            # prev_res=db.get_next_greater_row_by_roll_header_id(roll_id)
            

            image_paths = [msg[enums.IMAGE_PATH_COLUMN] for msg in roll_msgs]

            results = model.process_images(image_paths)

            combined_results = []

            # for idx, msg in enumerate(roll_msgs):
            #     image_path = msg[enums.IMAGE_PATH_COLUMN]
            #     combined = {
            #         enums.IMAGE_TITLE_COLUMN: os.path.basename(image_path),
            #         enums.IMAGE_PATH_COLUMN: image_path,
            #         enums.METERS_COLUMN: msg.get(enums.METERS_COLUMN),
            #         enums.ROLL_HEADER_ID_COLUMN: roll_id,
            #         enums.IS_BARCODE_READABLE_COLUMN: msg.get(enums.IS_BARCODE_READABLE_COLUMN)
            #         # "seam_data": seam_result_map.get(image_path)
            #     }
            
            #     combined_results.append(combined)
            success=db.insert_inference_results_bulk(results)

            if success and delivery_tags:
                rmq.acknowledge_messages(delivery_tags)
            with open("results.json", "w", encoding="utf-8") as f:
                json.dump({"images": results}, f, indent=2, ensure_ascii=False)



            # Sort QR candidates by meters before sending to QR model

            # Run QR detection

            # save_results_to_json({"images": combined_results}, f"results_{roll_id}.json")
# 
            # all_combined_results.extend(combined_results)

        return {"status": "success", "message": f"Processed {len(all_combined_results)} images across {len(roll_groups)} rolls"}

    except Exception as e:
        logging.error("Error in inference_images: %s", e)
        print(f"❌ Error in inference_images: {e}")
        return {"status": "error", "message": str(e)}
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)