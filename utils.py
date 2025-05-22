

import os
import os
import logging
import pandas as pd 
import enums
import json
def sanitize_qr_res(qr_results):
    try:
        sanitized_qr_results = []
        for item in qr_results:
            print("Checking item:", item)
            if isinstance(item, dict):
                sanitized_qr_results.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        sanitized_qr_results.append(parsed)
                except json.JSONDecodeError:
                    continue  # or log and skip
        return sanitized_qr_results
    except Exception as e:
        print(f"❌ Error sanitizing QR results: {e}")
def get_barcode_from_object(obj):
        qr        = obj.get("qr_data") or {}

        ocr       = qr.get("ocr") or {}
        barcode_from_json=""
        plain_text = (ocr.get("Plain_text") or "").strip()
        fast_qr=(qr.get("fast_qr")      or "").strip()
        low_confidence_matches = ""
        if fast_qr:
            barcode_from_json = fast_qr
        else:
            # Fallback to similarity_test -> matches[0] -> match
            similarity_test = qr.get("similarity_test") or {}
            matches = similarity_test.get("matches") or []

            if matches and isinstance(matches, list) and len(matches) > 0 and "match" in matches[0]:
                # Check if score is at least 0.9
                if "score" in matches[0] and matches[0]["score"] >= 0.9:
                    barcode_from_json = matches[0]["match"]
                else:
                    # Score is less than 0.9 - use original barcode from OCR
                    barcode_from_json = ocr.get("Plain_text", "")  # use plain_text as original barcode
                    low_confidence_matches = matches  # Save matches for reference
                    print(f"Low confidence match ({matches[0].get('score', 0)}) - using original barcode")
            else:
                barcode_from_json = plain_text  # default if all sources fail
        return barcode_from_json
def normalize_barcode(code):
    """Normalize barcode for more reliable comparison"""
    if not code:
        return ""
    # Remove all whitespace
    code = "".join(code.split())
    # Ensure consistent case (less relevant for numeric codes)
    code = code.upper()
    return code    
def get_barcode_position(combined_results, target_barcode):

    try:
        # Initialize flag to track if a seam has been found
        is_seam_found = False
        
        # Loop through all results
        for obj in combined_results:
            # Check if this frame has a seam
            seam_data = obj.get("seam_data", {})
            if seam_data and seam_data.get("final_class") == "seam":
                is_seam_found = True
            
            # Check if this frame has the target barcode
            barcode=get_barcode_from_object(obj)
            if normalize_barcode(barcode)==normalize_barcode(target_barcode):
                if is_seam_found:
                    return 1
                return 2
                
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error in get_barcode_position: {e}")
        print(f"❌ Error in get_barcode_position: {e}")
        return 0
def check_and_add_seams_between_barcodes(combined_results):
    """
    Identifies barcodes in the results, checks if seams exist between consecutive barcodes,
    and inserts synthetic seams where needed.
    
    Args:
        combined_results: List of processed image results with seam and QR data
        
    Returns:
        tuple: (seams_added, updated_results)
            seams_added: Number of synthetic seams added
            updated_results: The original or modified combined_results
    """
    try:
        # Find all barcodes and their positions
        barcode_positions = []
        
        for i, result in enumerate(combined_results):
            barcode = get_barcode_from_object(result)
            if barcode:  # If barcode exists
                barcode_positions.append({
                    "index": i,
                    "barcode": barcode,
                    "meter": result.get(enums.METERS_COLUMN),
                    "path": result.get(enums.IMAGE_PATH_COLUMN)
                })
        
        if len(barcode_positions) < 2:
            print(f"Found {len(barcode_positions)} barcodes - need at least 2 to check for seams")
            return 0, combined_results
            
        print(f"Found {len(barcode_positions)} barcodes")
        
        # Sort by meter position to ensure correct order
        barcode_positions.sort(key=lambda x: x["meter"] if x["meter"] is not None else float('inf'))
        
        # Check for seams between consecutive barcode pairs
        updated_results = combined_results.copy()
        seams_added = 0
        
        for i in range(len(barcode_positions) - 1):
            pos1 = barcode_positions[i]
            pos2 = barcode_positions[i + 1]
            
            idx1, idx2 = pos1["index"], pos2["index"]
            
            # Check if there's a seam between these barcodes
            seam_exists = False
            for j in range(min(idx1, idx2), max(idx1, idx2) + 1):
                seam_data = updated_results[j].get("seam_data", {})
                if seam_data and seam_data.get("final_class") == "seam":
                    seam_exists = True
                    print(f"✅ Seam found between {pos1['barcode']} and {pos2['barcode']}")
                    break
            
            # If no seam exists, create a synthetic one
            if not seam_exists:
                # Calculate average meter position
                if pos1["meter"] is not None and pos2["meter"] is not None:
                    avg_meter = (pos1["meter"] + pos2["meter"]) / 2
                else:
                    # Insert at middle index if meter info not available
                    avg_meter = None
                
                # Create synthetic seam entry
                roll_header_id = combined_results[idx1].get(enums.ROLL_HEADER_ID_COLUMN)
                synthetic_seam = {
                    enums.IMAGE_TITLE_COLUMN: f"synthetic_seam_{seams_added}",
                    enums.IMAGE_PATH_COLUMN: f"synthetic_path_{seams_added}",
                    enums.METERS_COLUMN: avg_meter,
                    enums.ROLL_HEADER_ID_COLUMN: roll_header_id,
                    enums.IS_BARCODE_READABLE_COLUMN: False,
                    "seam_data": {
                        "final_class": "seam",
                        "conf": 0.95  # High confidence for synthetic seam
                    },
                    "synthetic": True,  # Mark as synthetic entry
                    "created_between": {
                        "barcode1": pos1["barcode"],
                        "barcode2": pos2["barcode"]
                    }
                }
                
                # Find position to insert based on meter value
                insert_pos = idx1 + 1
                if avg_meter is not None:
                    while insert_pos < len(updated_results):
                        current_meter = updated_results[insert_pos].get(enums.METERS_COLUMN)
                        if current_meter is not None and current_meter > avg_meter:
                            break
                        insert_pos += 1
                
                # Insert synthetic seam
                updated_results.insert(insert_pos, synthetic_seam)
                seams_added += 1
                
                print(f"⚠️ No seam found between {pos1['barcode']} and {pos2['barcode']} - inserted synthetic seam at {avg_meter} meters")
                
                # Update indexes of barcode positions after this insertion
                for k in range(i + 1, len(barcode_positions)):
                    if barcode_positions[k]["index"] >= insert_pos:
                        barcode_positions[k]["index"] += 1
        
        print(f"Added {seams_added} synthetic seams")
        return seams_added, updated_results
        
    except Exception as e:
        logging.error(f"❌ Error in check_and_add_seams_between_barcodes: {e}")
        print(f"❌ Error in check_and_add_seams_between_barcodes: {e}")
        return 0, combined_results
def apply_checks(combined_results, prev_df, raw_messages,roll_id):
    try:
        is_update_prev_result = False
        updated_prev_rows = []
        is_seam,combined_results=check_and_add_seams_between_barcodes(combined_results)
        # prev_df = 

        # # Prepare previous results DataFrame
        # if all_prev_results_df is None or all_prev_results_df.empty or 'roll_header_id' not in all_prev_results_df.columns:
        #     prev_df = pd.DataFrame().rename_axis('roll_header_id')
        # else:
        #     prev_df = all_prev_results_df.set_index('roll_header_id')

        # Collect records and delivery tags
        records_to_insert = []
        delivery_tags = []

        latest_is_piece_completed=0
        #1 if piece is completed, 0 otherwise
        for i, result in enumerate(combined_results):
            try:
                # Ack tag
                method_frame = raw_messages[i]["method_frame"]
                delivery_tags.append(method_frame.delivery_tag)

                # Defaults

                latest_barcode          = ""
                latest_box_coord        = None
                latest_best_box_coord   = None
                barcode_manual= ""
                is_fastqr=0
                latest_is_barcode_model = 0 #1 if barcode model is detected, 0 otherwise
                latest_is_barcode_manual= 0 #1 if barcode is detected manually like if the prevous was empty
                latest_is_sticker_model = 0#1 if sticker model is detected, 0 otherwise
                latest_is_sticker_manual= 0#1 if sticker is detected manually like if the prevous was empty

                # roll_header_id = result.get(enums.ROLL_HEADER_ID_COLUMN) #Trolley ID of the roll    
                # prev_res = (
                #     prev_df.loc[[roll_id]]
                #     if roll_id in prev_df.index
                #     else pd.DataFrame()
                # )
                
                prev_res = prev_df
                qr        = result.get("qr_data") or {}
                detection = qr.get("detection") or {}
                ocr       = qr.get("ocr") or {}

                # pull and trim both sources
                barcode_from_json=""
                plain_text = (ocr.get("Plain_text") or "").strip()
                fast_qr=(qr.get("fast_qr")      or "").strip()
                low_confidence_matches = ""
                if fast_qr:
                    is_fastqr=1
                    barcode_from_json = fast_qr
                else:
                    # Fallback to similarity_test -> matches[0] -> match
                    similarity_test = qr.get("similarity_test") or {}
                    matches = similarity_test.get("matches") or []

                    if matches and isinstance(matches, list) and len(matches) > 0 and "match" in matches[0]:
                        # Check if score is at least 0.9
                        if "score" in matches[0] and matches[0]["score"] >= 0.7:
                            barcode_from_json = matches[0]["match"]
                            
                            print(f"High confidence match ({matches[0].get('score', 0)}) - using barcode from similarity test")
                            print("Barcode from similarity test:", barcode_from_json)
                        else:
                            # Score is less than 0.9 - use original barcode from OCR
                            barcode_from_json = ocr.get("Plain_text", "")  # use plain_text as original barcode
                            low_confidence_matches = matches  # Save matches for reference
                            print(f"Low confidence match ({matches[0].get('score', 0)}) - using original barcode")
                    else:
                        barcode_from_json = plain_text  # default if all sources fail
                # detection true iff either is non-empty
                has_detection = bool(fast_qr) or bool(plain_text)
                # print('Pre. Res: ',prev_res)
                condition_col=""

                if result.get(enums.IS_BARCODE_READABLE_COLUMN) and barcode_from_json:
                    print('Roll header id:',roll_id)
                    print('Pre. Res: ',prev_res)                    
                    latest_is_sticker_model=1
                    # pick whichever value is available
                    # latest_barcode = fast_qr or (plain_text.split()[0] if plain_text else "") #Current barcode value                    latest_barcode      = barcode_value 
                    latest_barcode = barcode_from_json
                    latest_box_coord    = detection.get("box") or {} #Coordinates of the barcode
                    latest_best_box_coord = detection.get("best_box") or {} #Coordinates of the best box
                    # is_second,seam=check_is_barcode_second(latest_barcode,roll_header_id)
                    if not prev_res.empty: #Barocde is second
                        if normalize_barcode(prev_res.iloc[0].get("barcode_model")) == normalize_barcode(latest_barcode):
                            latest_is_barcode_model = 1
                            latest_is_barcode_model=1
                            latest_is_piece_completed=1
                            prev_df=pd.DataFrame()
                            print(f"At if prev_res.iloc[0].get(barcode_model) == latest_barcode")
                            print('Barcode: ',latest_barcode)
                            condition_col="At if prev_res.iloc[0].get(barcode_model) == latest_barcode"
                        else:
                            # latest_barcode=prev_res.iloc[0].get("barcode_model")
                            prev_is_fastqr=prev_res.iloc[0].get("is_fastqr")
                            if prev_is_fastqr==1 and is_fastqr==0:
                                barcode_manual=prev_res.iloc[0].get("barcode_model")
                                latest_barcode=prev_res.iloc[0].get("barcode_model")
                                latest_is_barcode_manual=1
                                latest_is_sticker_model=1
                                latest_is_piece_completed=1
                                

                            elif is_fastqr==1 and prev_is_fastqr==0:
                                is_update_prev_result = True
                                index = prev_res.index[0]  # first row
                                prev_res.at[index, 'barcode_manual'] = latest_barcode
                                prev_res.at[index, 'is_sticker_model'] = 1
                                prev_res.at[index,'is_barcode_manual']=1
                                prev_res.at[index,'barcode_model']=''
                                prev_res.at[index,'is_barcode_model']=0
                                updated_prev_rows.append(prev_res.loc[index].copy())
                                latest_is_piece_completed=1
                            else:
                                latest_barcode=prev_res.iloc[0].get("barcode_model")                                
                                # latest_is_barcode_model = 0
                                latest_is_barcode_manual=1
                                latest_is_sticker_model=1
                                latest_is_piece_completed=1                                
                                                                                                                            
                            prev_df=pd.DataFrame()
                            print(f"At Else of  if prev_res.iloc[0].get(barcode_model) == latest_barcode")
                            print('Barcode: ',latest_barcode)
                            condition_col="At Else of  if prev_res.iloc[0].get(barcode_model) == latest_barcode"
                    elif get_barcode_position(combined_results, latest_barcode) == 2:
                        print("At elif get_barcode_position(combined_results, latest_barcode) == 2")
                        current_meter = result.get(enums.METERS_COLUMN, 0)
                        print('Current meter:', current_meter)
                        
                        # Get records from DB that might need updating
                        second_barcode_df = db.is_barcode_second(roll_id, current_meter)
                        print('Second barcode df:', second_barcode_df)
                        
                        # Set these flags regardless of DB state
                        latest_is_barcode_model = 1
                        latest_is_sticker_model = 1
                        latest_is_piece_completed=1
                        
                        if not second_barcode_df.empty:
                            index = second_barcode_df.index[0]  # first row
                            
                            # Check if there's already a barcode present
                            existing_barcode_manual = second_barcode_df.loc[index, 'barcode_manual']
                            existing_barcode_model = second_barcode_df.loc[index, 'barcode_model']
                            existing_barcode_meter=second_barcode_df.loc[index,'meters']
                            existing_barcode = existing_barcode_manual or existing_barcode_model
                            
                            if existing_barcode and str(existing_barcode).strip():
                                print(f"Found existing barcode '{existing_barcode}', creating new entry for '{latest_barcode}'")
                                
                                # Create a new entry for insertion rather than updating
                                new_record = {
                                    enums.ROLL_HEADER_ID_COLUMN: roll_id,
                                    enums.IMAGE_TITLE_COLUMN: 'dummy.jpg',
                                    enums.IMAGE_PATH_COLUMN: 'dummy_path',
                                    "image_in_cm": 0.0,
                                    "image_class_id": 'normal',
                                    "confidence": result.get("seam_data", {}).get("conf"),
                                    "box_coord": latest_box_coord,
                                    "best_box_coord": latest_best_box_coord,
                                    "is_sticker_model": 1,
                                    "is_sticker_manual": 0,
                                    "is_barcode_manual": 1,
                                    "is_barcode_model": 0,
                                    "is_damage_barcode": 0,
                                    "barcode_manual": latest_barcode,
                                    "barcode_model": latest_barcode,
                                    "plain_text": plain_text,
                                    enums.METERS_COLUMN: existing_barcode_meter+0.5,
                                    "is_piece_completed": 0,
                                    "low_confidence_matches": low_confidence_matches,
                                    "condition_col": "New entry for existing barcode"
                                }
                                
                                # Add to records to insert
                                records_to_insert.append(new_record)
                        
                    else:
                        latest_is_piece_completed=0
                        print("At Else")
                        print('Barcode: ',latest_barcode)
                        print("Latest box coord:", latest_box_coord)
                        print("Latest best box coord:", latest_best_box_coord)
                        condition_col="At Else"

                    
                    # elif is_second:
                        # update the seam
                
                # build the DB record
                db_record = {
                    enums.ROLL_HEADER_ID_COLUMN: roll_id,
                    enums.IMAGE_TITLE_COLUMN:    os.path.basename(result.get(enums.IMAGE_TITLE_COLUMN)),
                    enums.IMAGE_PATH_COLUMN:     result.get(enums.IMAGE_PATH_COLUMN),
                    "image_in_cm":               0.0,
                    "image_class_id":            result.get("seam_data", {}).get("final_class"),
                    "confidence":                result.get("seam_data", {}).get("conf"),
                    "box_coord":                 latest_box_coord,
                    "best_box_coord":            latest_best_box_coord,
                    "is_sticker_model":          latest_is_sticker_model,
                    "is_sticker_manual":         latest_is_sticker_manual,
                    "is_barcode_manual":         latest_is_barcode_manual,
                    "is_barcode_model":          latest_is_barcode_model,
                    "is_damage_barcode":         0,
                    "barcode_manual":            barcode_manual,
                    "barcode_model":             latest_barcode,
                    "plain_text":                plain_text,
                    enums.METERS_COLUMN:         result.get(enums.METERS_COLUMN, 0),
                    "is_piece_completed":      latest_is_piece_completed,
                    "low_confidence_matches": low_confidence_matches,
                    "condition_col": condition_col,
                    "is_fastqr": is_fastqr
                }
                records_to_insert.append(db_record)
                latest_is_piece_completed=0
                # lates

            except Exception as e:
                logging.error("❌ Failed to prepare record for %s: %s",
                              result.get(enums.IMAGE_PATH_COLUMN), e)

        # Bulk insert all records
        if records_to_insert:
            try:
                print("Bulk inserting records...")
                if db.insert_inference_results_bulk(records_to_insert):
                    print("✅ Bulk insert successful")
                    if delivery_tags:
                        last_tag = max(delivery_tags)
                        rmq.channel.basic_ack(delivery_tag=last_tag, multiple=True)
                        print(f"✅ Acknowledged {len(delivery_tags)} messages")
                else:
                    logging.error("❌ Bulk insert failed - likely due to empty roll_header_id")
                    # ack only those with empty header
                    empty_tags = [
                        t for rec, t in zip(records_to_insert, delivery_tags)
                        if not rec.get(enums.ROLL_HEADER_ID_COLUMN)
                    ]
                    if empty_tags:
                        last_tag = max(empty_tags)
                        rmq.channel.basic_ack(delivery_tag=last_tag, multiple=True)
                        print(f"✅ Ack {len(empty_tags)} msgs with empty roll_header_id")

            except Exception as e:
                logging.error("❌ Bulk insert error: %s", e)

        # Update any modified “previous” rows
        if is_update_prev_result and updated_prev_rows:
            try:
                db.close()
                db.connect()
                updated_df = pd.DataFrame(updated_prev_rows)
                db.update_previous_detection_from_df(updated_df)
                print("✅ Updated previous result(s) in DB")
            except Exception as e:
                logging.error("❌ Failed to update previous results: %s", e)

    except Exception as e:
        logging.error("❌ Exception in apply_checks: %s", e)
        print(f"❌ Exception in apply_checks: {e}")
