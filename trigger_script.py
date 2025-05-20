import requests
import time

def trigger_inference():
    url = "http://127.0.0.1:8000/inference_image"
    
    while True:
        try:
            response = requests.get(url)
            response.raise_for_status()
            print({"status": "success", "message": response.json()})
        except requests.exceptions.RequestException as e:
            print({"status": "error", "message": str(e)})

        # Optional: Wait 1 second before the next request to avoid overwhelming the server
        time.sleep(1)

# Start the loop
if __name__ == "__main__":
    trigger_inference()
