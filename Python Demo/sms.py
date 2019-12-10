url = "https://www.fast2sms.com/dev/bulk"
     
payload = "sender_id=FSTSMS&message=AccidentWitnessedAtTheLocation:https://www.google.com/maps/place/Indian+Institute+of+Technology+Bombay/@19.1334353,72.9110739,17z/data=!4m5!3m4!1s0x3be7c7f189efc039:0x68fdcea4c5c5894e!8m2!3d19.1334302!4d72.9132679&language=english&route=p&numbers=7355780958"
headers = {
 'authorization': "wvJesBUI4mGl5AYpkEDjr1q9ZaFcz2oOChS3RXdtfixMVTHLQbxqUplsf9IAibQ23yzBVt6RCwkehjDS",
 'Content-Type': "application/x-www-form-urlencoded",
 'Cache-Control': "no-cache",
}
     
response = requests.request("POST", url, data=payload, headers=headers)
     
print(response.text)
