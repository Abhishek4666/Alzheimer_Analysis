from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.append('~/airflow/dags/')  # Change to your script directory

from alzheimers_pipeline import run_pyspark_job  # Import the function

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),  # Adjust as needed
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'alzheimers_ml_pipeline',
    default_args=default_args,
    description='A PySpark ML pipeline for Alzheimer diagnosis',
    schedule_interval=None,  # Run manually
    catchup=False
)

# Define the task
run_pyspark_task = PythonOperator(
    task_id='run_pyspark_script',
    python_callable=run_pyspark_job,
    dag=dag
)

# Set task dependencies (if needed)
run_pyspark_task
