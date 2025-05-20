import os
import json
import pymysql
import pandas as pd
import enums
import uuid
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",         # Log output goes to this file
    filemode="a"                # Append to the file (use "w" to overwrite each time)
)

class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        self.local_path="D:\EVS"
        self.path_to_change="\\172.16.52.214"

    def connect(self):
        """Establish connection to the database."""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                db=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connected to database successfully")
        except Exception as e:
            print(f"❌ Exception at <DBConnection>: {e}")
            return False
        return True

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.commit()
            self.cursor.close()
            self.connection.close()
            print("✅ Database connection closed.")
   

    def insert_inference_results_bulk(self, data_source):
        """
        Bulk insert inference results from a JSON file or directly from a data list.
        Each result gets a unique UUID, uses the image path from the data,
        and stores the complete image data as JSON in predictedAi.
        
        Args:
            data_source: Either a string path to a JSON file or a list of result dictionaries
        """
        try:
            # Handle both file path and direct data input
            if isinstance(data_source, str):
                # It's a file path
                with open(data_source, 'r') as file:
                    results = json.load(file)
            else:
                # Assume it's already the results list
                results = data_source
            
            # Prepare SQL query
            sql = f"""
                INSERT INTO {enums.ROLL_BODY_COLUMN} (
                    imageId, imagePath, predictedAi   
                ) VALUES (
                    %s, %s, %s
                )
            """
            
            # Prepare values for bulk insert
            values = []
            for result in results:
                # Generate a unique UUID for each image
                image_id = str(uuid.uuid4()).replace('-', '')[:31]
                
                # Get the image path from the result
                image_path = result.get('image_path', '')
                
                # Use the complete result JSON as predictedAi
                predicted_ai_json = json.dumps(result)
                
                values.append((
                    image_id,
                    image_path,
                    predicted_ai_json
                ))
            
            # Execute the bulk insert
            self.cursor.executemany(sql, values)
            self.connection.commit()
            logging.info(f"✅ Bulk inserted {len(values)} inference results")
            print(f"✅ Bulk inserted {len(values)} inference results")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error inserting bulk inference results: {e}")
            print(f"❌ Error inserting bulk inference results: {e}")
            return False
    def insert_inference_result(self, result: dict):
        """
        Inserts a structured inference result into the database.
        """
        try:
            sql = f"""
                INSERT INTO {enums.ROLL_BODY_COLUMN} (
                    image_title, image_in_cm, image_path, image_class_id, confidence,
                    box_coord, best_box_coord, is_sticker_model, is_sticker_manual,
                    is_barcode_manual, barcode_manual, barcode_model, meters,roll_header_id,is_damage_barcode,col_1
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,%s,%s,
                )
            """
            values = (
                result.get(enums.IMAGE_TITLE_COLUMN),
                result.get("image_in_cm"),
                result.get(enums.IMAGE_PATH_COLUMN),
                result.get("image_class_id"),
                result.get("confidence"),
                json.dumps(result.get("box_coord")) if result.get("box_coord") else None,
                json.dumps(result.get("best_box_coord")) if result.get("best_box_coord") else None,
                int(result.get("is_sticker_model", 0)),
                int(result.get("is_sticker_manual", 0)),
                int(result.get("is_barcode_manual", 0)),
                int(result.get("is_damage_barcode")),

                result.get("barcode_manual"),
                result.get("barcode_model"),
                result.get(enums.METERS_COLUMN),
                result.get(enums.ROLL_HEADER_ID_COLUMN),
                result.get("col_1")
                
                
                # result.get("col_1")
                # result.get(enums.IS_BARCODE_READABLE_COLUMN)
            )

            self.cursor.execute(sql, values)
            self.connection.commit()
            print(f"✅ Inserted inference result for: {result.get('image_title')}")
        except Exception as e:
            logging.error(f"❌ Error inserting inference result: {e}")
            print(f"❌ Error inserting inference result: {e}")

    def insert_data(self,json_score):
        try:

            sql = """
                INSERT INTO inference_output ( result)
                VALUES ( %s)
            """
            values = ( json.dumps(json_score))

            # Insert into MySQL
            try:
                self.cursor.execute(sql, values)
                self.connection.commit()
                print(f"✅ Inserted: ")
            except Exception as e:
                print(f"❌ Error inserting {e}")
        except Exception as e:
            print('Exception at <insert_data>',str(e))




# if __name__ == "__main__":
#     jsonn={
#     "image": "0001_000_00.png",
#     "model1_prediction": "seam",
#     "model1_confidence": 83.9,
#     "model2_prediction": "seam",
#     "model2_confidence": 92.82,
#     "timestamp": "2025-02-28T10:50:17.892934"
# }
#     # Database credentials
#     db_config = {
#         'host': 'localhost',
#         'user': 'root',
#         'password': 'edra@k',
#         'database': 'fault_app'
#     }


#     # Create the database handler
#     db_handler = DatabaseHandler(
#         host=db_config['host'],
#         user=db_config['user'],
#         password=db_config['password'],
#         database=db_config['database']
#     )

#     # Connect to the database
#     if db_handler.connect():
#         # Create the image data inserter
#         print(db_handler.fetch_data())
#         # db_handler.insert_data("image.jpg","ffsasdd",jsonn)

#         # Close the database connection
#         db_handler.close()
#     else:
#         print("❌ Database connection failed.")
