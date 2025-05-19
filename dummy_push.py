import os
import json
import pika
import random
from pathlib import Path

# Configuration
IMAGE_FOLDER = "seam_images/barcode-test-1"  # Folder containing images to be pushed to RabbitMQ
RABBITMQ_HOST = "localhost"
RABBITMQ_QUEUE = "evs_images_input_queue"
import enums
def get_image_files(folder):
    # Acceptable image extensions
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    return [f for f in Path(folder).glob("*") if f.suffix.lower() in exts]

def push_images_to_queue():
    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()

    # Ensure queue exists
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

    image_files = get_image_files(IMAGE_FOLDER)
    meter=1
    for img in image_files:
        
        message = {
            # enums.IMAGE_TITLE_COLUMN: img.stem,
            enums.IMAGE_PATH_COLUMN: str(img.resolve()),
            enums.ROLL_HEADER_ID_COLUMN: 101348404067565620,
            enums.METERS_COLUMN:meter,  # Random meters for demo purposes
            enums.IS_BARCODE_READABLE_COLUMN: True  # Assuming barcode is readable for demo
            # You can change this logic
        }
        meter+=1

        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2  # make message persistent
            )
        )
        print(f"✅ Sent: {message}")

    connection.close()
    print("📦 All images pushed to queue.")

if __name__ == "__main__":
    push_images_to_queue()
