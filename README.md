# mnist-training-errors

### Install steps
```
$ pip install gspread==3.6.0
$ pip install oauth2client
```

Get 'client_secret.json' (originally got via https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html) from Google if you want to use Google Sheets addition (and then adapt the spreadsheet code for your spreadsheet layout)

### Running
```
$ time python mnist_training_errors.py
  ...
  -----------
  Last element:  9999
  Total errors:  219
  
  real	0m55.439s
  user	1m42.566s
  sys	0m7.245s
```
On my dual core macbook which indicates GPU usage of TensorFlow since 'user' time is 3 times 'clock' time
