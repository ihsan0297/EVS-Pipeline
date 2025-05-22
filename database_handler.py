import os
import json
import pymysql
import pandas as pd
import enums

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
    def fetch_barcodes(self):
        """
        Fetches the latest records from table_6 where co_32 is not null,
        and returns them as a list of tuples: [(id, path), ...]
        """
        try:
            sql = """
            SELECT DISTINCT col_32  
            FROM table_6 
            ORDER BY U_TIME DESC ;
            
            """
            self.cursor.execute(sql)

            # Fetch results and normalize paths
            self.cursor.execute(sql)
            results = [row[0] for row in self.cursor.fetchall()]
            return results
        except Exception as e:
            logging.error(f"❌ Error fetching data: {e}")
            print('Exception at <fetch_barcodes>', str(e))
            return []

    def update_previous_detection_from_df(self,df):
        try:
            if df.empty:
                print("❌ DataFrame is empty. Nothing to update.")
                return

            row = df.iloc[0].to_dict()
            record_id = row.get('id')

            if record_id is None:
                print("❌ 'id' column not found in DataFrame.")
                return

            # Remove 'id' from update fields
            update_fields = {k: v for k, v in row.items() if k != 'id'}
            set_clause = ", ".join([f"{k} = %s" for k in update_fields.keys()])
            values = list(update_fields.values()) + [record_id]

            query = f"""
            UPDATE {enums.ROLL_BODY_COLUMN}
            SET {set_clause}
            WHERE id = %s
            """

            # with db_conn.cursor() as cursor:
            self.cursor.execute(query, values)
            self.connection.commit()


            print(f"✅ Record updated for id = {record_id}")

        except Exception as e:
            logging.error(f"❌ Error updating record with id = {record_id}: {e}")
            print(f"❌ Error updating record with id = {record_id}: {e}")

    def update_ids(self, ids):
        try:
            if not ids:
                print("⚠️ No IDs provided for update.")
                return

            # Create placeholders for query (e.g., if 3 IDs: "?, ?, ?")
            placeholders = ", ".join(["%s"] * len(ids))
            sql = f"""
            UPDATE fault_app.roll_body
            SET col_10 = '1'
            WHERE id IN ({placeholders})
            """
            
            # Execute query with provided IDs
            self.cursor.execute(sql, tuple(ids))
            self.connection.commit()
            
            print(f"✅ Updated {len(ids)} rows successfully")
        except Exception as e:
            print(f"❌ Error updating IDs: {e}")
    def fetch_data(self):
        try:
            sql = """
            SELECT id, col_7 
            FROM fault_app.roll_body 
            WHERE col_7 <> ''  
             ORDER BY U_TIME DESC
            LIMIT 10;
            """
            self.cursor.execute(sql)

            # Fetch results and process paths
            data = [
                (row[0], os.path.normpath(row[1].replace(self.path_to_change, self.local_path)).replace("\\", "/").lstrip("/"))  
                for row in self.cursor.fetchall()
            ]
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["id", "path"])
            
            return df
        except Exception as e:
            logging.error(f"❌ Error fetching data: {e}")
            print('Exception at <fetch_data>', str(e))
            return pd.DataFrame(columns=["id", "path"])  # Return empty DataFrame on failure

            return []
    def is_barcode_second(self, roll_header_id,start_meters):
        try:
            query = f"CALL is_barcode_second_test({roll_header_id},{start_meters})"
            df = pd.read_sql_query(query, self.connection)
            return df

        except Exception as e:
            logging.error(f"❌ Error fetching data: {e}")
            print('Exception at <fetch_data>', str(e))
            return pd.DataFrame(columns=["id", "path"])

    #If this return empty row it means barcode is first vice versa
    def fetch_barcode_by_header_id(self, roll_header_id):
        try:
            query = f"CALL get_barcodes_by_header_id({roll_header_id})"
            df = pd.read_sql_query(query, self.connection)
            barcode_array = df.values.flatten().tolist()
            return barcode_array

        except Exception as e:
            logging.error(f"❌ Error fetching data: {e}")
            print('Exception at <fetch_data>', str(e))
            return pd.DataFrame(columns=["id", "path"])

    def get_next_greater_row_by_roll_header_id(self, roll_header_id):
        try:
            query = f"CALL get_next_greater_row_by_roll_header_id_test({roll_header_id})"
            df = pd.read_sql_query(query, self.connection)
            return df

        except Exception as e:
            logging.error(f"❌ Error fetching data: {e}")
            print('Exception at <fetch_data>', str(e))
            return pd.DataFrame(columns=["id", "path"])

    def fetch_latest_sticker_model(self, roll_header_id):
        """
        Fetches the latest record from evs.evs_roll_body where is_sticker_model = 1 
        for a specific roll_header_id.
        """
        try:
            query = f"""
                SELECT * FROM evs.{enums.ROLL_BODY_COLUMN}
                WHERE is_sticker_model = 1 AND roll_header_id = %s
                ORDER BY meters DESC
                LIMIT 1;
            """
            self.cursor.execute(query, (roll_header_id,))
            result = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(result, columns=columns)

            if not df.empty:
                print("✅ Latest sticker model record fetched.")
            else:
                print("⚠️ No record found with is_sticker_model = 1 for roll_header_id =", roll_header_id)
            return df

        except Exception as e:
            logging.error(f"❌ Error fetching latest sticker model: {e}")
            print(f"❌ Error fetching latest sticker model: {e}")
            return None

    def insert_inference_results_bulk(self, results: list):
        try:
                """
                Bulk insert multiple inference results using executemany.
                """
                sql = f"""
                    INSERT INTO {enums.ROLL_BODY_COLUMN} (
                        image_title, image_in_cm, image_path, image_class_id, confidence,
                        box_coord, best_box_coord, is_sticker_model, is_sticker_manual,
                        is_barcode_manual, barcode_manual, barcode_model, meters,
                        roll_header_id, col_1,is_piece_completed,col_2,col_3,is_fastqr
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                """
                values = []
                for r in results:
                    
                    low_confidence = r.get("low_confidence_matches")
                    if isinstance(low_confidence, (list, dict)):
                        low_confidence_json = json.dumps(low_confidence)
                    else:
                        low_confidence_json = low_confidence                    
                    
                    
                    values.append((
                        r.get(enums.IMAGE_TITLE_COLUMN),
                        r.get("image_in_cm"),
                        r.get(enums.IMAGE_PATH_COLUMN),
                        r.get("image_class_id"),
                        r.get("confidence"),
                        json.dumps(r.get("box_coord")) if r.get("box_coord") else json.dumps([]),
                        json.dumps(r.get("best_box_coord")) if r.get("best_box_coord") else json.dumps([]),
                        int(r.get("is_sticker_model", 0)),
                        int(r.get("is_sticker_manual", 0)),
                        int(r.get("is_barcode_manual", 0)),
                        r.get("barcode_manual"),
                        r.get("barcode_model"),
                        r.get(enums.METERS_COLUMN),
                        r.get(enums.ROLL_HEADER_ID_COLUMN),
                        r.get("plain_text"),
                        r.get("is_piece_completed", 0),
                        low_confidence_json,
                        r.get("condition_col"),
                        r.get("is_fastqr", 0)
                        # r.get("is_barcode_model")
                    ))
                self.cursor.executemany(sql, values)
                self.connection.commit()
        except Exception as e:
            print(f"❌ Error inserting bulk inference results: {e}")
            return False
        return True

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
