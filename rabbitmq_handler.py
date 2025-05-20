import pika
import json
import asyncio
import enums
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",         # Log output goes to this file
    filemode="a"                # Append to the file (use "w" to overwrite each time)
)
class RMQHandler:
    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        # self.database = database
        self.channel=None
        self.max_msgs=10
        
    def rabbitMQConnection(self):
        try:
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(self.host, credentials=credentials)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            self.channel=channel
            return True
        except Exception as e:
            print('Exception at <rabbitMQConnection>'+str(e))
            return False
    async def test_rmq(self,data):
        print(data)

    def fetch_messages(self):
        aggregated = {
            "images": [],
            "raw": []
        }
        for _ in range(self.max_msgs):
            method_frame, header_frame, body = self.channel.basic_get(queue=enums.RMQ_QUEUE_NAME, auto_ack=False)
            if method_frame:
                try:
                    data = json.loads(body.decode())

                    # Only append if 'image_path' is present
                    image_path = data.get(enums.IMAGE_PATH_COLUMN)
                    # image_title = data.get(enums.IMAGE_TITLE_COLUMN)
                    header_id = data.get(enums.ROLL_HEADER_ID_COLUMN)
                    is_barcode_readable=data.get(enums.IS_BARCODE_READABLE_COLUMN)
                    meters = data.get(enums.METERS_COLUMN)

                    if image_path:
                        aggregated["images"].append({
                            # enums.IMAGE_TITLE_COLUMN: image_title,
                            enums.IMAGE_PATH_COLUMN: image_path,
                            enums.ROLL_HEADER_ID_COLUMN: header_id,
                            enums.IS_BARCODE_READABLE_COLUMN: is_barcode_readable,
                            enums.METERS_COLUMN: meters
                        })
                        
                        aggregated["raw"].append({
                        "method_frame": method_frame,
                        "body": body
                    })

                        # Acknowledge the message only if processed successfully
                        # self.channel.basic_ack(method_frame.delivery_tag)

                except Exception as e:
                    print(f"❌ Error decoding message: {e}")
                    # Optionally reject message without requeueing
                    self.channel.basic_nack(method_frame.delivery_tag, requeue=False)
            else:
                break  # No more messages in queue

        return aggregated
    
    def callback(self ,method, body,i):
        try:
            data = json.loads(body.decode())
            # Process and store the data in SQL Server
            # with DBConnection() as connection:
            #     store_data_in_sql(data, connection)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.test_rmq(data))
    #        asyncio.get_event_loop().run_until_complete(send_data_to_remote_websocket(data))
            #store_data_in_sql(data)
            self.channel.basic_ack(delivery_tag=method.delivery_tag)
            return True
        except Exception as e:
            print('Exception at <callBack>'+str(e))
            logging.error(f"Error processing message: {e}")
            return False
    def acknowledge_messages(self, raw_messages):

        try:
            """
            Acknowledge multiple messages efficiently by extracting delivery tags
            and using the highest tag with multiple=True parameter.
            
            Args:
                raw_messages: List of raw message dictionaries containing method_frame
            """
            if not raw_messages:
                logging.info("No messages to acknowledge")
                return
            
            # Extract delivery tags from the raw message dictionaries
            delivery_tags = []
            for msg in raw_messages:
                if isinstance(msg, dict) and 'method_frame' in msg:
                    method_frame = msg.get('method_frame')
                    if method_frame and hasattr(method_frame, 'delivery_tag'):
                        delivery_tags.append(method_frame.delivery_tag)
            
            if not delivery_tags:
                logging.info("No valid delivery tags found in messages")
                return
                
            # Find the maximum delivery tag
            max_tag = max(delivery_tags)
            
            # Acknowledge all messages up to and including the max_tag
            self.channel.basic_ack(delivery_tag=max_tag, multiple=True)
            logging.info(f"✅ Acknowledged {len(delivery_tags)} messages (up to tag {max_tag})")
            print(f"✅ Acknowledged {len(delivery_tags)} messages (up to tag {max_tag})")
        except Exception as e:
            logging.error(f"❌ Error acknowledging messages: {e}")
            print(f"❌ Error acknowledging messages: {e}")
            return False
#MainMethod

# The code block you provided is the main part of the script. Here's what it does:
# if __name__ == "__main__":
    
#     rmq=RMQHandler('localhost', 'root','edraak123')
#     if rmq.rabbitMQConnection():
#         print('Connection Success')
#     else:
#         print('Connection Failed')
#         exit()
#     # rmq.rabbitMQConnection()
#     while True:
#         method_frame, header_frame, body = rmq.channel.basic_get(queue='evs_images_input_queue', auto_ack=False)
#         if not method_frame is None:
#         #     break  # No more messages in the queue
#             rmq.callback( method_frame, body,1)
