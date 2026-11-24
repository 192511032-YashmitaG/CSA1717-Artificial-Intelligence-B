CSA1717 ARTIFICIAL INTELLIGENCE – ASSIGNMENT 5 (CO5)
Implementation package

Contents
- src/assignment5_co5.py: modular drone, expert-system (including run-time rule update and regression helpers), intrusion-model, chatbot and optional PySpark RDD implementation.
- results/results.json: experiment outputs generated for the accompanying report.
- results/summary.csv: concise metric summary.
- evidence/: charts used in the report.
- README.txt: run instructions and scope notes.

Environment
Python 3.10+
Required: numpy, scikit-learn, matplotlib, joblib
Optional: pyspark (required only to execute the true Spark RDD pipeline)

Run
python -m pip install numpy scikit-learn matplotlib joblib
python src/assignment5_co5.py --out results

PySpark RDD pipeline
Install pyspark separately, prepare a text file with sensorId,timestamp,value per line,
then import pyspark_sensor_pipeline from src/assignment5_co5.py and call it with the file path.
The assignment environment used to produce this package did not have PySpark installed.
Accordingly, the report labels the 10^5/10^6/10^7 partition-scaling measurements as a local
NumPy aggregation proxy, NOT Spark execution timings. The actual Spark implementation is supplied.

Evidence integrity and assumptions
- Drone fleet tests are grid-world simulations with fixed seeds; no real drones or radio links were used.
- Pump cases are a small synthetic expert-labelled set. The first 15 cases are the development
  set and the last 15 are the hold-out set; rule performance should not be interpreted as field accuracy.
- Intrusion classification uses a generated, labelled benchmark created with scikit-learn make_classification,
  not a public operational intrusion dataset. The split is stratified and fixed at 75:25.
- Edge compression is represented by reducing tree count and maximum depth in a Random Forest.
- Chatbot examples are a small authored intent corpus and simulated word corruption.
- Latency and model-size results are machine/environment dependent. Re-run on target hardware before deployment.
