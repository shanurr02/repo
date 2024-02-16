import requests
import json

def make_api_request(text_data, image_file_path):
    # Flask API endpoint URL
    api_url = "http://127.0.0.1:5000/predict"

    # Prepare data
    text_payload = {'text': json.dumps(text_data)}
    files = {'image': ('image_file.png', open(image_file_path, 'rb'))}

    # Make the request
    response = requests.post(api_url, data=text_payload, files=files)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return {'error': f'Request failed with status code {response.status_code}'}

if __name__ == "__main__":

    text_data = [{'id': 1, 'text': 'Bank Offer10% instant discount on ICICI Bank Credit Cards, up to ₹300, on orders of ₹2,500 and aboveT&C'},
                 {'id': 2, 'text': 'Women Green Hand-held Bag - Extra Spacious  (Pack of: 5)'}]
    image_file_path = 'image.png'

    result = make_api_request(text_data, image_file_path)
    print(result)
