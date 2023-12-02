import requests

# code to download the model
def download_file(url):
    print("downloading model...")
    filename = 'car_logo_classifier_model.h5'
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

# url of model (uploaded to firebase storage)
url = "https://firebasestorage.googleapis.com/v0/b/scrapper-6e7db.appspot.com/o/car_logo_classifier_model.h5?alt=media&token=0eb9bf60-ce7c-426c-bcf5-9f0566d79d90"
download_file(url)
