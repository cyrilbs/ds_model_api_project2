import requests
import json
import argparse
import time 

# purpose:
#
# - go through all models as listed in 'models' and check score is above 'minScore'
# - produce a csv file given that header is model_name,testing_label,test_status
# - option --address is for testing server or container, default is to target api in a docker compose context
# - requests waits a few seconds for the api to be ready, this is useful with docker compose

# functions

def score_test(model):
  url="http://{address}:{port}/{model}/score".format(address=api_address, port=api_port, model=model)

  r = requests.get(url, time.sleep(3))
  
  return r

# main

parser = argparse.ArgumentParser(description='score test')
parser.add_argument('--address', '-a', nargs=1, type=str, help="address")
args = parser.parse_args()

if  args.address:
  api_address   = args.address[0]
  log_file_name = "api_test.log"
else:
  api_address   = 'api_reviews'
  log_file_name = "/app/api_test.log"
   
api_port      = 5000
result        = ""
minScore      = 0.6

models = ["all_branches"]

for model in models:

  r = score_test(model)

  score = json.loads(r.text)["score"]
  if score < minScore:
      status = "failed"
  else:
      status = "success"

  result+=model+",score,"+status

with open(log_file_name, 'a') as file:
  file.write(result+"\n")

