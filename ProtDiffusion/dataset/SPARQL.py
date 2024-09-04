import requests
import time
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the SPARQL endpoint and query
endpoint = "https://sparql.uniprot.org/"
query_template = """
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX up: <http://purl.uniprot.org/core/>

SELECT DISTINCT ?clusterid ?proteinid ?familytaxonid ?sequence
WHERE {{
  VALUES ?familytaxon {{ taxon:2 taxon:2759 }}
  VALUES ?identity {{ 0.5 }}
  
  ?organism rdfs:subClassOf ?familytaxon .

  ?sequenceClass a up:Sequence ;
                 rdf:value ?sequence ;
                 up:memberOf ?cluster ;
                 up:sequenceFor ?protein .
                
  ?protein a up:Protein ;
                up:organism ?organism .
  
  ?cluster up:identity ?identity .

  BIND(substr(str(?familytaxon), 34) AS ?familytaxonid)
  BIND(substr(str(?cluster), 32) AS ?clusterid)
  BIND(substr(str(?protein), 33) AS ?proteinid)
                
  FILTER NOT EXISTS {{
    ?protein up:annotation ?annotation .
    ?annotation a ?annotationType .
    FILTER (?annotationType = up:Non-adjacent_Residues_Annotation || 
            ?annotationType = up:Non-terminal_Residue_Annotation)
  }}
  
  FILTER ( ! contains(?sequence, 'X') && ! contains(?sequence, 'B') && ! contains(?sequence, 'Z'))     
}}
LIMIT {limit} OFFSET {offset}
"""

# Function to execute the SPARQL query with retries
def execute_query(query, retries=3, delay=5, offset=0):
    headers = {
        "Accept": "text/csv",
        "User-Agent": "s204514@dtu.dk"
    }
    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=1200) # Increase timeout if needed, right now it is 20 minutes
            response.raise_for_status()
            logging.info(f"Query successful for offset {offset}")
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed for offset {offset}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Max retries reached for offset {offset}. Exiting.")
                return None

# Function to fetch data for a specific offset
def fetch_data(offset, limit):
    query = query_template.format(limit=limit, offset=offset)
    return execute_query(query, offset=offset)

# Pagination parameters
limit = 100000  # Adjust this value based on performance observations
offset = 0
max_workers = 10  # Number of parallel workers

# Open a file to write the results
with open("output.csv", "w") as f:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while True:
            # Submit tasks to fetch data in parallel
            for i in range(max_workers):
                current_offset = offset + i * limit
                logging.info(f"Submitting query for offset {current_offset}")
                futures.append(executor.submit(fetch_data, current_offset, limit))
                time.sleep(1)  # Add a delay to avoid overloading the server
            
            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    break
                
                # Write the result to the file
                if offset == 0:
                    f.write(result)  # Write header for the first chunk
                else:
                    f.write(result.split("\n", 1)[1])  # Skip header for subsequent chunks
                
                # Update offset for the next chunk
                offset += limit
            
            # Clear the futures list for the next batch
            futures.clear()
            
            # Check if we have fetched all data
            if len(result.splitlines()) <= 1:
                break

logging.info("Data download complete.")