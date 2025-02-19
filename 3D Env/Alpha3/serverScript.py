import json
import time
import requests

# The server URL providing the camera data
SERVER_URL = "http://falsafa.in:8081/camera/1"  # Replace with actual server URL

# Path to the camera1.json file
CAMERA_JSON_PATH = "camera1.json"

# Function to fetch camera data from the server
def fetch_camera_data():
    try:
        response = requests.get(SERVER_URL)
        response.raise_for_status()  # Raises HTTPError if the response code is 4xx/5xx
        return response.json()  # Assuming the server responds with JSON data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from server: {e}")
        return None

# Function to update the camera1.json file with new data
def update_camera_json(data):
    try:
        with open(CAMERA_JSON_PATH, 'w') as json_file:
            data = data["frames"][-1]
            json.dump(data, json_file, indent=4)
        print("camera1.json updated successfully.")
    except IOError as e:
        print(f"Error writing to file: {e}")

# Main loop to keep updating camera1.json every second
def main():
    while True:
        camera_data = fetch_camera_data()
        
        if camera_data:
            update_camera_json(camera_data)
        
        # Wait for 1 second before fetching the next update
        time.sleep(1)

if __name__ == "__main__":
    main()
