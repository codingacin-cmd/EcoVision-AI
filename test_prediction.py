
import requests

try:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={
            "vegetation": 73,
            "rainfall": 77,
            "soil_quality": 77,
            "human_activity": 16
        },
        timeout=30
    )

    print("Status:", response.status_code)
    print("Response:", response.text)

except requests.RequestException as error:
    print("Connection error:", error)
