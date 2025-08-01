import requests

form_data = {
    "TYPE": "250+10(028)"
}
response = requests.post("http://localhost:5000/predict", data=form_data)
print(response.json())
