#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl, os
from urllib import request
from urllib.error import HTTPError
from time import sleep


BASE_URL_EUK = "https://www.ebi.ac.uk/interpro/wwwapi/protein/UniProt/taxonomy/uniprot/2" # Prokaryotic
BASE_URL_PRO = "https://www.ebi.ac.uk/interpro/wwwapi/protein/UniProt/taxonomy/uniprot/2759" # Eukaryotic

# phosphopantehteine attachment, ACP
URL_EXTENSION = "/entry/InterPro/IPR009081/?page_size=200&extra_fields=sequence"

OUTPUT_FILE = "ACP_by_IPR.csv"

def output_list(output_handle, url, familytaxonid):
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = url
  last_page = False

  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        # no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      entries = None
      if ("entry_subset" in item):
        entries = item["entry_subset"]
      elif ("entries" in item):
        entries = item["entries"]
      
      seq = item["extra_fields"]["sequence"]
      output_handle.write(item["metadata"]["accession"] + "," + familytaxonid + "," + seq + "\n")
    
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(1)

if __name__ == "__main__":

  if os.path.isfile(OUTPUT_FILE):
    print(OUTPUT_FILE + " already exists, please remove or rename it before running this script")
    sys.exit(errno.EEXIST)
  else:
    open(OUTPUT_FILE, 'a').close()
  
  with open(OUTPUT_FILE, "a") as output_handle:
    output_handle.write("proteinid,familytaxonid,sequence\n")

    url = BASE_URL_EUK + URL_EXTENSION
    output_list(output_handle, url, familytaxonid="2759")
    url = BASE_URL_PRO + URL_EXTENSION
    output_list(output_handle, url, familytaxonid="2")