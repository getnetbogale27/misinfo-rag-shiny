# Service wrapper for calling the Python FastAPI backend.

library(httr)
library(jsonlite)

analyze_claim <- function(claim) {
  # Send claim to backend and parse JSON response.
  response <- POST(
    url = "http://localhost:8000/analyze",
    body = list(claim = claim),
    encode = "json"
  )

  # Raise an error if the API call failed.
  stop_for_status(response)

  # Convert response body to an R list.
  content(response, as = "parsed", type = "application/json")
}

run_model_evaluation <- function(dataset_path = "data/evaluation_dataset.json") {
  response <- POST(
    url = "http://localhost:8000/evaluate",
    body = list(dataset_path = dataset_path),
    encode = "json"
  )

  stop_for_status(response)
  content(response, as = "parsed", type = "application/json")
}
